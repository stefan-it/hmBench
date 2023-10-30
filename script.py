# Script for HF autotrain space runner 🚀
# Expected environment variables:
# CONFIG: points to *.json configuration file
# HF_TOKEN: HF access token from https://huggingface.co/settings/tokens
# REPO_NAME: name of HF datasets repo
import os
import flair
import json
import importlib

from huggingface_hub import login, HfApi

from pathlib import Path

fine_tuner = importlib.import_module("flair-fine-tuner")

config_file = os.environ.get("CONFIG")
hf_token = os.environ.get("HF_TOKEN")
hf_hub_org_name = os.environ.get("HUB_ORG_NAME")

login(token=hf_token, add_to_git_credential=True)
api = HfApi()

with open(config_file, "rt") as f_p:
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
                    fine_tuner.run_experiment(seed, batch_size, epoch, learning_rate, subword_pooling, hipe_datasets,
                                              json_config)
                    dataset_identifier = hipe_datasets[0] if len(hipe_datasets) == 1 else "mhmner"
                    hf_model = json_config["hf_model"]
                    context_size = json_config["context_size"]
                    layers = json_config["layers"] if "layers" in json_config else "-1"
                    use_crf = json_config["use_crf"] if "use_crf" in json_config else False
                    use_tensorboard_logger = json_config[
                        "use_tensorboard_logger"] if "use_tensorboard_logger" in json_config else False

                    if context_size == 0:
                        context_size = False

                    # configs/newseye/fr/hmbyt5.json -> hmbyt5
                    hf_model_short = config_file.split("/")[-1].replace(".json", "")

                    repo_name = f'hmbench-{dataset_identifier.replace("/", "-")}-{hf_model_short}-bs{batch_size}-ws{context_size}-e{epoch}-lr{learning_rate}-pooling{subword_pooling}-layers{layers}-crf{use_crf}-{seed}'
                    output_path = f"hmbench-{dataset_identifier}-{hf_model}-bs{batch_size}-ws{context_size}-e{epoch}-lr{learning_rate}-pooling{subword_pooling}-layers{layers}-crf{use_crf}-{seed}"

                    repo_url = api.create_repo(
                        repo_id=f"{hf_hub_org_name}/{repo_name}",
                        token=hf_token,
                        private=True,
                        exist_ok=True,
                    )

                    if use_tensorboard_logger:
                        api.upload_folder(
                            folder_path=f"{output_path}/runs",
                            path_in_repo="./runs",
                            repo_id=f"{hf_hub_org_name}/{repo_name}",
                            repo_type="model"
                        )

                    best_model_test_path = Path(f"{output_path}/best-model.pt")
                    best_model_name = "best-model.pt"

                    if not best_model_test_path.exists():
                        # In some rare cases no best model was written (e.g. when F1-score is 0 for all epochs)
                        best_model_name = "final-model.pt"

                    api.upload_file(
                        path_or_fileobj=f"{output_path}/{best_model_name}",
                        path_in_repo="./pytorch_model.bin",
                        repo_id=f"{hf_hub_org_name}/{repo_name}",
                        repo_type="model"
                    )
                    api.upload_file(
                        path_or_fileobj=f"{output_path}/training.log",
                        path_in_repo="./training.log",
                        repo_id=f"{hf_hub_org_name}/{repo_name}",
                        repo_type="model"
                    )
