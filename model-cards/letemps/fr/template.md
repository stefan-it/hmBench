---
base_model: {BASE_MODEL}
tags:
  - flair
  - token-classification
  - sequence-tagger-model
language: fr
widget:
  - text: "Parmi les remèdes recommandés par la Société , il faut mentionner celui que M . Schatzmann , de Lausanne , a proposé :"
license: mit
---

# Fine-tuned Flair Model on LeTemps French NER Dataset (HIPE-2022)

This Flair model was fine-tuned on the
[LeTemps French](https://github.com/hipe-eval/HIPE-2022-data/blob/main/documentation/README-letemps.md)
NER Dataset using {BASE_MODEL_SHORT} as backbone LM.

The LeTemps dataset consists of NE-annotated historical French newspaper articles from mid-19C to mid 20C.

The following NEs were annotated: `loc`, `org` and `pers`.