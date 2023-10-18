---
base_model: {BASE_MODEL}
tags:
  - flair
  - token-classification
  - sequence-tagger-model
language: fr
widget:
  - text: "Nous recevons le premier numéro d ' un nouveau journal , le Radical - Libéral , qui paraîtra à Genève deux fois la semaine . Son but est de représenter l ' élément national du radicalisme genevois , en d ' autres termes , de défendre la politique intransigeante do M . Carteret , en opposition aux tendances du groupe _ > dont le Genevois est l ' organe . Bétail ."
license: mit
---

# Fine-tuned Flair Model on French HIPE-2020 Dataset (HIPE-2022)

This Flair model was fine-tuned on the
[French HIPE-2020](https://github.com/hipe-eval/HIPE-2022-data/blob/main/documentation/README-hipe2020.md)
NER Dataset using {BASE_MODEL_SHORT} as backbone LM.

The HIPE-2020 dataset is comprised of newspapers from mid 19C to mid 20C. For information can be found
[here](https://dl.acm.org/doi/abs/10.1007/978-3-030-58219-7_21).

The following NEs were annotated: `loc`, `org`, `pers`, `prod`, `time` and `comp`.