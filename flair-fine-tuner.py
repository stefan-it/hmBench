import json
import logging
import sys
import flair

from flair import set_seed

from typing import List

from flair.data import MultiCorpus
from flair.datasets import NER_HIPE_2022, NER_ICDAR_EUROPEANA
from flair.embeddings import TransformerWordEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from flair.trainers.plugins.loggers.tensorboard import TensorboardLogger

from pathlib import Path

from byt5_embeddings import ByT5Embeddings

from hisgermaner_dataset import HisGermaNER

from utils import prepare_ajmc_corpus, prepare_clef_2020_corpus, prepare_newseye_fi_sv_corpus, prepare_newseye_de_fr_corpus

logger = logging.getLogger("flair")
logger.setLevel(level="INFO")


def run_experiment(seed: int, batch_size: int, epoch: int, learning_rate: float, subword_pooling: str,
                   hipe_datasets: List[str], json_config: dict):
    hf_model = json_config["hf_model"]
    context_size = json_config["context_size"]
    layers = json_config["layers"] if "layers" in json_config else "-1"
    use_crf = json_config["use_crf"] if "use_crf" in json_config else False
    label_name_map = json_config["label_name_map"] if "label_name_map" in json_config else None
    use_tensorboard_logger = json_config["use_tensorboard_logger"] if "use_tensorboard_logger" in json_config else False

    # Set seed for reproducibility
    set_seed(seed)

    corpus_list = []

    # Dataset-related
    for dataset in hipe_datasets:
        dataset_name, language = dataset.split("/")

        # E.g. topres19th needs no special preprocessing
        preproc_fn = None

        if dataset_name == "ajmc":
            preproc_fn = prepare_ajmc_corpus
        elif dataset_name == "hipe2020":
            preproc_fn = prepare_clef_2020_corpus
        elif dataset_name == "newseye" and language in ["fi", "sv"]:
            preproc_fn = prepare_newseye_fi_sv_corpus
        elif dataset_name == "newseye" and language in ["de", "fr"]:
            preproc_fn = prepare_newseye_de_fr_corpus

        if dataset_name == "icdar":
            corpus_list.append(NER_ICDAR_EUROPEANA(language=language))
        elif dataset_name == "hisgermaner":
            corpus_list.append(HisGermaNER())
        else:
            corpus_list.append(NER_HIPE_2022(dataset_name=dataset_name, language=language, preproc_fn=preproc_fn,
                                             label_name_map=label_name_map, add_document_separator=True))

    if context_size == 0:
        context_size = False

    logger.info("FLERT Context: {}".format(context_size))
    logger.info("Layers: {}".format(layers))
    logger.info("Use CRF: {}".format(use_crf))

    corpora: MultiCorpus = MultiCorpus(corpora=corpus_list, sample_missing_splits=False)
    label_dictionary = corpora.make_label_dictionary(label_type="ner")
    logger.info("Label Dictionary: {}".format(label_dictionary.get_items()))

    embeddings = None

    if "byt5" in hf_model:
        logger.info("Using own implementation of ByT5Embeddings")
        embeddings = ByT5Embeddings(
            model=hf_model,
            layers=layers,
            subword_pooling=subword_pooling,
            fine_tune=True,
        )
    else:
        embeddings = TransformerWordEmbeddings(
            model=hf_model,
            layers=layers,
            subtoken_pooling=subword_pooling,
            fine_tune=True,
            use_context=context_size,
        )

    tagger: SequenceTagger = SequenceTagger(
        hidden_size=256,
        embeddings=embeddings,
        tag_dictionary=label_dictionary,
        tag_type="ner",
        use_crf=use_crf,
        use_rnn=False,
        reproject_embeddings=False,
    )

    # Trainer
    trainer: ModelTrainer = ModelTrainer(tagger, corpora)

    dataset_identifier = hipe_datasets[0] if len(hipe_datasets) == 1 else "mhmner"

    output_path = f"hmbench-{dataset_identifier}-{hf_model}-bs{batch_size}-ws{context_size}-e{epoch}-lr{learning_rate}-pooling{subword_pooling}-layers{layers}-crf{use_crf}-{seed}"

    plugins = []

    if use_tensorboard_logger:
        logger.info("TensorBoard logging is enabled")

        tb_path = Path(f"{output_path}/runs")
        tb_path.mkdir(parents=True, exist_ok=True)

        plugins.append(TensorboardLogger(log_dir=str(tb_path), comment=output_path))

    trainer.fine_tune(
        output_path,
        learning_rate=learning_rate,
        mini_batch_size=batch_size,
        max_epochs=epoch,
        shuffle=True,
        embeddings_storage_mode='none',
        weight_decay=0.,
        use_final_model_for_eval=False,
        plugins=plugins,
    )

    # Finally, print model card for information
    tagger.print_model_card()


if __name__ == "__main__":
    filename = sys.argv[1]
    with open(filename, "rt") as f_p:
        json_config = json.load(f_p)

    seeds = json_config["seeds"]
    batch_sizes = json_config["batch_sizes"]
    epochs = json_config["epochs"]
    learning_rates = json_config["learning_rates"]
    subword_poolings = json_config["subword_poolings"]

    hipe_datasets = json_config["hipe_datasets"]  # Do not iterate over them

    cuda = json_config["cuda"]
    flair.device = f'cuda:{cuda}'

    for seed in seeds:
        for batch_size in batch_sizes:
            for epoch in epochs:
                for learning_rate in learning_rates:
                    for subword_pooling in subword_poolings:
                        run_experiment(seed, batch_size, epoch, learning_rate, subword_pooling, hipe_datasets,
                                       json_config)  # pylint: disable=no-value-for-parameter
