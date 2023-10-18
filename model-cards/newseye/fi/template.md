---
base_model: {BASE_MODEL}
tags:
  - flair
  - token-classification
  - sequence-tagger-model
language: fi
widget:
  - text: Rooseveltin sihteeri ilmoittaa perättö - mäksi tiedon , että Rooseveltia olisi kehotettu käymään Englannissa , Saksassa ja Venäjällä puhumassa San Franciscon näyttelyn puolesta .
license: mit
---

# Fine-tuned Flair Model on Finnish NewsEye NER Dataset (HIPE-2022)

This Flair model was fine-tuned on the
[Finnish NewsEye](https://github.com/hipe-eval/HIPE-2022-data/blob/main/documentation/README-newseye.md)
NER Dataset using {BASE_MODEL_SHORT} as backbone LM.

The NewsEye dataset is comprised of diachronic historical newspaper material published between 1850 and 1950
in French, German, Finnish, and Swedish.
More information can be found [here](https://dl.acm.org/doi/abs/10.1145/3404835.3463255).

The following NEs were annotated: `PER`, `LOC`, `ORG` and `HumanProd`.