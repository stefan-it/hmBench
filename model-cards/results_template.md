# Results

We performed a hyper-parameter search over the following parameters with 5 different seeds per configuration:

* Batch Sizes: {BATCH_SIZES}
* Learning Rates: {LEARNING_RATES}

And report micro F1-score on development set:

{RESULTS}

The [training log](training.log) and TensorBoard logs (not available for hmBERT Base model) are also uploaded to the model hub.

More information about fine-tuning can be found [here](https://github.com/stefan-it/hmBench).
