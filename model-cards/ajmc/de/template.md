---
base_model: {BASE_MODEL}
tags:
  - flair
  - token-classification
  - sequence-tagger-model
language: de
widget:
  - text: — Dramatiſch war der Stoff vor Sophokles von Äſchylos behandelt worden in den Θροῇσσαι , denen vielleicht in der Trilogie das Stüc>"OnJw» κοίσις vorherging , das Stück Σαλαμίνιαι folgte .
license: mit
---

# Fine-tuned Flair Model on AjMC German NER Dataset (HIPE-2022)

This Flair model was fine-tuned on the
[AjMC German](https://github.com/hipe-eval/HIPE-2022-data/blob/main/documentation/README-ajmc.md)
NER Dataset using {BASE_MODEL_SHORT} as backbone LM.

The AjMC dataset consists of NE-annotated historical commentaries in the field of Classics,
and was created in the context of the [Ajax MultiCommentary](https://mromanello.github.io/ajax-multi-commentary/)
project.

The following NEs were annotated: `pers`, `work`, `loc`, `object`, `date` and `scope`.