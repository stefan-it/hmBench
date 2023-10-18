---
base_model: {BASE_MODEL}
tags:
  - flair
  - token-classification
  - sequence-tagger-model
language: fr
widget:
  - text: Je suis convaincu , a-t43 dit . que nous n"y parviendrions pas , mais nous ne pouvons céder parce que l' état moral de nos troupe* en souffrirait trop . ( Fournier . ) Des avions ennemis lancent dix-sept bombes sur Dunkerque LONDRES . 31 décembre .
license: mit
---

# Fine-tuned Flair Model on French ICDAR-Europeana NER Dataset

This Flair model was fine-tuned on the
[French ICDAR-Europeana](https://github.com/stefan-it/historic-domain-adaptation-icdar)
NER Dataset using {BASE_MODEL_SHORT} as backbone LM.

The ICDAR-Europeana NER Dataset is a preprocessed variant of the
[Europeana NER Corpora](https://github.com/EuropeanaNewspapers/ner-corpora) for Dutch and French.

The following NEs were annotated: `PER`, `LOC` and `ORG`.