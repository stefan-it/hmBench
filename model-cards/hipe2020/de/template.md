---
base_model: {BASE_MODEL}
tags:
  - flair
  - token-classification
  - sequence-tagger-model
language: de
widget:
  - text: "Es war am 25sten , als Lord Corn wollis Dublin mit seinem Gefolge und mehrern Truppen verließ , um in einer Central - Lage bey Sligo die Operationen der Armee persönlich zu dirigiren . Der Feind dürfte bald in die Enge kommen , da Gen . Lacke mit 6000 Mann ihm entgegen marschirt ."
license: mit
---

# Fine-tuned Flair Model on German HIPE-2020 Dataset (HIPE-2022)

This Flair model was fine-tuned on the
[German HIPE-2020](https://github.com/hipe-eval/HIPE-2022-data/blob/main/documentation/README-hipe2020.md)
NER Dataset using {BASE_MODEL_SHORT} as backbone LM.

The HIPE-2020 dataset is comprised of newspapers from mid 19C to mid 20C. For information can be found
[here](https://dl.acm.org/doi/abs/10.1007/978-3-030-58219-7_21).

The following NEs were annotated: `loc`, `org`, `pers`, `prod`, `time` and `comp`.