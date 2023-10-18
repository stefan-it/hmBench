# hmBench: A Benchmark for Historical Language Models on NER Datasets

![hmBench](cute_library_sheep.jpeg)

This repository presents a benchmark for Historical Language Models with main focus on NER Datasets such as
[HIPE-2022](https://github.com/hipe-eval/HIPE-2022-data/tree/main).

## Models

The following Historical Language Models are currently used in benchmarks:

| Model   | Hugging Face Model Hub Org                                                                            |
|---------|-------------------------------------------------------------------------------------------------------|
| hmBERT  | [Historical Multilingual Language Models for Named Entity Recognition](https://huggingface.co/hmbert) |
| hmTEAMS | [Historical Multilingual TEAMS Models](https://huggingface.co/hmteams)                                |
| hmByT5  | [Historical Multilingual and Monolingual ByT5 Models](https://huggingface.co/hmbyt5)                  |

## Datasets

We benchmark pretrained language models on various datasets from HIPE-2020, HIPE-2022 and Europeana. The following table
shows an overview of used datasets:

| Language | Datasets                                                         |
|----------|------------------------------------------------------------------|
| English  | [AjMC] - [TopRes19th]                                            |
| German   | [AjMC] - [NewsEye] - [HIPE-2020]                                 |
| French   | [AjMC] - [ICDAR-Europeana] - [LeTemps] - [NewsEye] - [HIPE-2020] |
| Finnish  | [NewsEye]                                                        |
| Swedish  | [NewsEye]                                                        |
| Dutch    | [ICDAR-Europeana]                                                |

[AjMC]: https://github.com/hipe-eval/HIPE-2022-data/blob/main/documentation/README-ajmc.md
[NewsEye]: https://github.com/hipe-eval/HIPE-2022-data/blob/main/documentation/README-newseye.md
[TopRes19th]: https://github.com/hipe-eval/HIPE-2022-data/blob/main/documentation/README-topres19th.md
[ICDAR-Europeana]: https://github.com/stefan-it/historic-domain-adaptation-icdar
[LeTemps]: https://github.com/hipe-eval/HIPE-2022-data/blob/main/documentation/README-letemps.md
[HIPE-2020]: https://github.com/hipe-eval/HIPE-2022-data/blob/main/documentation/README-hipe2020.md

## Results

The `hmLeaderboard` space on the Hugging Face Model Hub shows all results and can be
accessed [here](https://huggingface.co/spaces/stefan-it/hmLeaderboard).

## Fine-Tuning

We use Flair for fine-tuning NER models on [HIPE-2022](https://github.com/hipe-eval/HIPE-2022-data) datasets from
[HIPE-2022 Shared Task](https://hipe-eval.github.io/HIPE-2022/). Additionally, the
[ICDAR-Europeana](https://github.com/stefan-it/historic-domain-adaptation-icdar) is used for benchmarks on Dutch and
French.

We use a tagged version of Flair to ensure a kind of reproducibility. The following commands need to be run to install
all necessary dependencies:

```bash
$ pip3 install -r requirements.txt
```

In order to use the hmTEAMS models you need to authorize with your account on Hugging Face Model Hub. This can be done
via cli:

```bash
# Use access token from https://huggingface.co/settings/tokens
$ huggingface-cli login
```

We use a config-driven hyper-parameter search. The script [`flair-fine-tuner.py`](flair-fine-tuner.py) can be used to
fine-tune NER models from our Model Zoo.

Additionally, we provide a script that uses Hugging Face
[AutoTrain Advanced (Space Runner)](https://github.com/huggingface/autotrain-advanced) to fine-tune models.
The following snippet shows an example:

```bash
$ pip3 install git+https://github.com/huggingface/autotrain-advanced.git
$  export HF_TOKEN="" # Get token from: https://huggingface.co/settings/tokens
$ autotrain spacerunner --project-name "flair-hmbench-hmbyt5-ajmc-de" \
  --script-path $(pwd) \
  --username stefan-it \
  --token $HF_TOKEN \
  --backend spaces-t4s \
  --env "CONFIG=configs/ajmc/de/hmbyt5.json;HF_TOKEN=$HF_TOKEN;HUB_ORG_NAME=stefan-it"
```

The concrete implementation can be found in [`script.py`](script.py).

**Notice**: the AutoTrain implementation is currently under development!

All configurations for fine-tuning are located in the `./configs` folder with the following naming convention:
`./configs/<dataset-name>/<language>/<model-name>.json`.

# Changelog

* 17.10.2023: 720 models from hyper-parameter search are now available on the [Model Hub](https://huggingface.co/stefan-it?search_models=hmbench).
* 05.10.2023: Initial version of this repository.

# Acknowledgements

We thank [Luisa März](https://github.com/LuisaMaerz), [Katharina Schmid](https://github.com/schmika) and
[Erion Çano](https://github.com/erionc) for their fruitful discussions about Historical Language Models.

Research supported with Cloud TPUs from Google's [TPU Research Cloud](https://sites.research.google/trc/about/) (TRC).
Many Thanks for providing access to the TPUs ❤️
