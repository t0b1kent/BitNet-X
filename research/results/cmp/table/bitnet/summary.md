Fact extraction exact: 572/585

| wording | query parse exact | hop1 | hop2 | hop3 | count | older | all |
|---|---|---|---|---|---|---|---|
| original | 150/150 | 29/30 | 28/30 | 30/30 | 29/30 | 30/30 | 97% [93-99] |
| paraphrase | 141/150 | 24/30 | 28/30 | 29/30 | 26/30 | 30/30 | 91% [86-95] |

Extraction errors (first 8):
* `Givedo owns a parrot.` -> `{"person": "Gideon", "attr": "pet", "value": "parrot"}`
* `Givedo lives in Jogipoton.` -> `{"person": "Gideon", "attr": "city", "value": "Jogipoton"}`
* `Givedo's mentor is Dove.` -> `{"person": "Giveno", "attr": "mentor", "value": "Dove"}`
* `Givedo is 66 years old.` -> `{"person": "Giveno", "attr": "age", "value": "66"}`
* `Givedo works as a doctor.` -> `{"person": "Gideon", "attr": "job", "value": "doctor"}`
* `Dodo lives in Habahaford.` -> `{"person": "Dodo", "attr": "city", "value": "Habahof"}`
* `Gilu works as a potter.` -> `{"person": "Gulu", "attr": "job", "value": "potter"}`
* `Gilu's mentor is Ferive.` -> `{"person": "Gulu", "attr": "mentor", "value": "Ferive"}`
