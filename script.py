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

fine_tuner = importlib.import_module("flair-fine-tuner")

config_file = os.environ.get("CONFIG")
hf_token = os.environ.get("HF_TOKEN")
repo_name = os.environ.get("REPO_NAME")

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
                    fine_tuner.run_experiment(seed, batch_size, epoch, learning_rate, subword_pooling, hipe_datasets, json_config)
                    api.upload_folder(
                        folder_path="./",
                        path_in_repo="./",
                        repo_id=repo_name,
                        repo_type="dataset",
                    )
