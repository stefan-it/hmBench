---
base_model: {BASE_MODEL}
tags:
  - flair
  - token-classification
  - sequence-tagger-model
language: en
widget:
  - text: Cp . Eur . Phoen . 240 , 1 , αἷμα ddiov φλέγέι .
license: mit
---

# Fine-tuned Flair Model on AjMC English NER Dataset (HIPE-2022)

This Flair model was fine-tuned on the
[AjMC English](https://github.com/hipe-eval/HIPE-2022-data/blob/main/documentation/README-ajmc.md)
NER Dataset using {BASE_MODEL_SHORT} as backbone LM.

The AjMC dataset consists of NE-annotated historical commentaries in the field of Classics,
and was created in the context of the [Ajax MultiCommentary](https://mromanello.github.io/ajax-multi-commentary/)
project.

The following NEs were annotated: `pers`, `work`, `loc`, `object`, `date` and `scope`.