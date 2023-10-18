---
base_model: {BASE_MODEL}
tags:
  - flair
  - token-classification
  - sequence-tagger-model
language: en
widget:
  - text: "On Wednesday , a public dinner was given by the Conservative Burgesses of Leads , to the Conservative members of the Leeds Town Council , in the Music Hall , Albion-street , which was very numerously attended ."
license: mit
---

# Fine-tuned Flair Model on TopRes19th English NER Dataset (HIPE-2022)

This Flair model was fine-tuned on the
[TopRes19th English](https://github.com/hipe-eval/HIPE-2022-data/blob/main/documentation/README-topres19th.md)
NER Dataset using {BASE_MODEL_SHORT} as backbone LM.

The TopRes19th dataset consists of NE-annotated historical English newspaper articles from 19C.

The following NEs were annotated: `BUILDING`, `LOC` and `STREET`.