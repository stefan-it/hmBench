---
base_model: {BASE_MODEL}
tags:
  - flair
  - token-classification
  - sequence-tagger-model
language: nl
widget:
  - text: Professoren der Geneeskun dige Faculteit te Groningen alsook van de HH , Doctoren en Chirurgijns van Groningen , Friesland , Noordholland , Overijssel , Gelderland , Drenthe , in welke Provinciën dit Elixir als Medicament voor Mond en Tanden reeds jaren bakend is .
license: mit
---

# Fine-tuned Flair Model on Dutch ICDAR-Europeana NER Dataset

This Flair model was fine-tuned on the
[Dutch ICDAR-Europeana](https://github.com/stefan-it/historic-domain-adaptation-icdar)
NER Dataset using {BASE_MODEL_SHORT} as backbone LM.

The ICDAR-Europeana NER Dataset is a preprocessed variant of the
[Europeana NER Corpora](https://github.com/EuropeanaNewspapers/ner-corpora) for Dutch and French.

The following NEs were annotated: `PER`, `LOC` and `ORG`.