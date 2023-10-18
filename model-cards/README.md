# Model Card Creation

In order to create model cards for over 720 models, we created base templates for each language of each dataset.

Once the Flair training logs are uploaded to the Model Hub, it is possible to create model cards including the results
for each hyper-parameter configuration and seed.

The `huggingface_hub` library is used and needs to be installed. Additionally, the user access token
(find it [here](https://huggingface.co/settings/tokens)) needs to be set as environment variable (`HF_TOKEN`):

```python
import os

import numpy as np

from collections import defaultdict
from huggingface_hub import login, HfApi, hf_hub_download, ModelCard
from tabulate import tabulate
from typing import List

hf_token = os.environ.get("HF_TOKEN")

login(token=hf_token, add_to_git_credential=True)
api = HfApi()
```

The following hyper-parameter are the same across all used backbone LMs (such as hmBERT, hmTEAMS or hmByT5):

```python
# Hyperparameters that are static across all experiments
context_size = "False"
epochs = "10"
pooling = "poolingfirst"
layers = "layers-1"
crf = "crfFalse"

batch_sizes = [8,4]
seeds = [1, 2, 3, 4, 5]
```

The following method parses the Flair training log and returns the best F1-score on development set:

```python
# Get best model score from training log
def get_best_model_score(input_file: str) -> float:
    all_dev_results = []
    with open(input_file, "rt") as f_p:
        for line in f_p:
            line = line.rstrip()
            if "f1-score (micro avg)" in line:
                dev_result = line.split(" ")[-1]
                all_dev_results.append(dev_result)

        return max([float(value) for value in all_dev_results])
```

Then the following method fetches all models for a dataset-language pair (e.g. hmBERT fine-tuned on German AjMC dataset)
and returns a nice results table:

```python
# Constructs results table
def get_results_table(dataset_name: str,
                      language: str,
                      model_name: str,
                      context_size: str,
                      epochs: str,
                      pooling: str,
                      layers: str,
                      crf: str,
                      batch_sizes: List[int],
                      learning_rates: List[float],
                      seeds: List[int]
                     ) -> str:

    # Configuration (= combination of batch size and learning rate) => [Best Model Score for each seed]
    dev_results = defaultdict(list)

    for batch_size in batch_sizes:
        for learning_rate in learning_rates:
            for seed in seeds:
                model_identifier = f"hmbench-{dataset_name}-{language}-{model_name}-bs{batch_size}-ws{context_size}-e{epochs}-lr{learning_rate}-{pooling}-{layers}-{crf}-{seed}"
                repo_id = f"stefan-it/{model_identifier}"
                current_repo = api.list_repo_refs(repo_id, repo_type="model")

                training_log_file = hf_hub_download(repo_id=repo_id, filename="training.log")

                best_model_score = get_best_model_score(training_log_file)
                dev_results[f"bs{batch_size}-e{epochs}-lr{learning_rate}"].append(best_model_score)
    
    # Configuration (= combination of batch size and learning rate) => Mean of all best dev scores
    mean_dev_results = {}

    for dev_result in dev_results.items():
        result_identifier, results = dev_result

        mean_result = np.mean([float(value) for value in results])

        mean_dev_results[result_identifier] = mean_result
    
    # Sorted dictionary (key -> best configuration)
    sorted_mean_dev_results = dict(sorted(mean_dev_results.items(), key=lambda item: item[1], reverse=True))

    best_dev_configuration = max(mean_dev_results, key=mean_dev_results.get)
    
    header = ["Configuration"] + [f"Run {i + 1}" for i in range(len(seeds))] + ["Avg."]

    # Sorted by best configuration, incl. link to models on the hub
    table = []

    ref_counter = 1
    ref_list = []

    for mean_dev_config, score in sorted_mean_dev_results.items():
        mean_dev_config_splitted = mean_dev_config.split("-")
        
        batch_size = mean_dev_config_splitted[0]
        epochs = mean_dev_config_splitted[1]
        
        if len(mean_dev_config_splitted) == 3:
            learning_rate = mean_dev_config_splitted[2]
        elif len(mean_dev_config_splitted) == 4:
            learning_rate = mean_dev_config_splitted[2] + "-" + mean_dev_config_splitted[3]

        model_identifier = f"hmbench-{dataset_name}-{language}-{model_name}-{batch_size}-ws{context_size}-{epochs}-{learning_rate}-{pooling}-{layers}-{crf}"
        repo_id = f"stefan-it/{model_identifier}"

        current_std = np.std(dev_results[mean_dev_config])
        current_row = [mean_dev_config]

        for seed in seeds:
            current_score = dev_results[mean_dev_config][seed - 1]
            current_row.append(f"[{current_score}][{ref_counter}]")

            ref_list.append(f"[{ref_counter}]: https://hf.co/{repo_id}-{seed}")

            ref_counter += 1


        current_row.append(f"{round(score * 100, 2)} ± {round(current_std * 100, 2)}")

        table.append(current_row)

    return tabulate(table, headers=header, tablefmt="github") + "\n\n" + "\n".join(ref_list)
```

The next function creates a `ModelCard` instance by reading and replacing all template variables:

```python
def get_model_card(dataset_name,
                   language,
                   base_model,
                   base_model_short,
                   batch_sizes,
                   learning_rates,
                   results_table):
    model_card = ""

    with open(f"./{dataset_name}/{language}/template.md", "rt") as f_p:
        for line in f_p:
            line = line.strip()

            model_card += line + "\n"

    model_card += "\n"

    # Only for hmByt5
    if base_model_short == "hmByT5":
        inference_notice = """# ⚠️ Inference Widget ⚠️

Fine-Tuning ByT5 models in Flair is currently done by implementing an own [`ByT5Embedding`][0] class.

This class needs to be present when running the model with Flair.

Thus, the inference widget is not working with hmByT5 at the moment on the Model Hub and is currently disabled.

This should be fixed in future, when ByT5 fine-tuning is supported in Flair directly.

[0]: https://github.com/stefan-it/hmBench/blob/main/byt5_embeddings.py
"""
        model_card += inference_notice + "\n"

        # Disable it
        model_card = model_card.replace("widget:", "inference: false\nwidget:")

    with open(f"./results_template.md", "rt") as f_p:
        for line in f_p:
            line = line.strip()

            model_card += line + "\n"

    model_card += "\n"

    with open(f"./acknowledgements_template.md", "rt") as f_p:
        for line in f_p:
            line = line.strip()

            model_card += line + "\n"

    model_card = model_card.replace("{BASE_MODEL}", base_model)
    model_card = model_card.replace("{BASE_MODEL_SHORT}", base_model_short)
    model_card = model_card.replace("{BATCH_SIZES}", f'`[{ ", ".join([str(bs) for bs in batch_sizes])}]`')
    model_card = model_card.replace("{LEARNING_RATES}", f'`[{ ", ".join([str(lr) for lr in learning_rates])}]`')
    model_card = model_card.replace("{RESULTS}", results_table)
    
    return ModelCard(model_card)
```

In the final step, the model cards are actually sent to the Model Hub for each model. This step needs to be done for
every backbone LM, because e.g. hmByT5 has different parameters.

For hmByT5, the model card creation for German HIPE-2020 can be started with:

```python
# Special for hmByT5
learning_rates = [0.00015, 0.00016]

base_model = "hmbyt5-preliminary/byt5-small-historic-multilingual-span20-flax"
base_model_short = "hmByT5"

model_name = "hmbyt5"

dataset_name = "hipe2020"
language = "de"

results_table = get_results_table(dataset_name,
                                  language,
                                  model_name,
                                  context_size,
                                  epochs,
                                  pooling,
                                  layers,
                                  crf,
                                  batch_sizes,
                                  learning_rates,
                                  seeds)

model_card = get_model_card(dataset_name,
                             language,
                             base_model,
                             base_model_short,
                             batch_sizes,
                             learning_rates,
                             results_table)

print(model_card)
```

It will then show a preview of a model card that.

The final commit is done via:

```python
for batch_size in batch_sizes:
    for learning_rate in learning_rates:
        for seed in seeds:
            model_identifier = f"hmbench-{dataset_name}-{language}-{model_name}-bs{batch_size}-ws{context_size}-e{epochs}-lr{learning_rate}-{pooling}-{layers}-{crf}-{seed}"
            repo_id = f"stefan-it/{model_identifier}"
            current_repo = api.list_repo_refs(repo_id, repo_type="model")
            
            print(f"Update Model Card PR for repo {repo_id}")
            model_card.push_to_hub(repo_id=repo_id,
                             create_pr=False,
                             commit_message="readme: fix link reference for ByT5 embedding implementation",
                            )
```

For hmBERT and hmTEAMS the procedure is pretty much the same, except the following parameters at the beginning:

```python
learning_rates = [3e-05, 5e-05]

base_model = "dbmdz/bert-base-historic-multilingual-cased"
base_model_short = "hmBERT"

model_name = "hmbert"
```

for hmBERT and these parameters for hmTEAMS:

```python
learning_rates = [3e-05, 5e-05]

base_model = "hmteams/teams-base-historic-multilingual-discriminator"
base_model_short = "hmTEAMS"

model_name = "hmteams"
```

# Model rename

By default, the Flair library stores a fine-tuned model under `best-model.pt`. However, the naming convention on the
Model Hub is `pytorch_model.bin`. Thus, renaming of the model is needed. This can be done with:

```python
import os

import os

from huggingface_hub import login, HfApi, plan_multi_commits, CommitOperationCopy, CommitOperationDelete

hf_token = os.environ.get("HF_TOKEN")

login(token=hf_token, add_to_git_credential=True)
api = HfApi()

# Hyperparameters that are static across all experiments
context_size = "False"
epochs = "10"
pooling = "poolingfirst"
layers = "layers-1"
crf = "crfFalse"

batch_sizes = [8,4]
seeds = [1, 2, 3, 4, 5]
```

For hmByT5 the following configuration is needed:

```python
learning_rates = [0.00015, 0.00016]

dataset_name = "topres19th"
language = "en"
model_name = "hmbyt5"
```

For hmBERT:

```python
learning_rates = [3e-05, 5e-05]

dataset_name = "topres19th"
language = "en"
model_name = "hmbert"
```

For hmTEAMS:

```python
learning_rates = [3e-05, 5e-05]

dataset_name = "newseye"
language = "de"
model_name = "hmteams"
```

Then the renaming can be started with:

```python
for batch_size in batch_sizes:
    for learning_rate in learning_rates:
        for seed in seeds:
            model_identifier = f"hmbench-{dataset_name}-{language}-{model_name}-bs{batch_size}-ws{context_size}-e{epochs}-lr{learning_rate}-{pooling}-{layers}-{crf}-{seed}"
            original_repo_id = f"stefan-it/{model_identifier}"

            print(f"Rename best-model.pt -> pytorch_model.bin for repo {original_repo_id}...")
            info = api.create_commit(repo_id=original_repo_id,
                                     commit_message="model: rename best-model.pt -> pytorch_model.bin",
                                     commit_description="This renames `best-model.pt` to `pytorch_model.bin` to fully support it as Flair model.",
                                     operations=[CommitOperationCopy(src_path_in_repo="best-model.pt", path_in_repo="pytorch_model.bin"), CommitOperationDelete(path_in_repo="best-model.pt")],
                                     create_pr=False)
```

# Model Visibility

By default, the models were uploaded with `private=True` attribute. To change the visibility to `public` the following

code can be run:

```python
from huggingface_hub import login, HfApi, move_repo, update_repo_visibility

for batch_size in batch_sizes:
    for learning_rate in learning_rates:
        for seed in seeds:
            model_identifier = f"hmbench-{dataset_name}-{language}-{model_name}-bs{batch_size}-ws{context_size}-e{epochs}-lr{learning_rate}-{pooling}-{layers}-{crf}-{seed}"
            original_repo_id = f"stefan-it/{model_identifier}"
            
            print(f"Update visibility to True for repo {original_repo_id}")
            update_repo_visibility(repo_id=original_repo_id, private=False)
```