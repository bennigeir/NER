# NER
T-725-MALV Final Project: 1.2 Named Entity Recognition (NER)

This repository is part of a final project in the course **T-725-MALV, Natural Language Processing** taught by Hrafn Loftsson and Hannes Högni Vilhjálmsson at Reykjavík University.
In this project a BERT model is applied to an Icelandic NER corpus. The model is evaluated using 10-fold cross-validation, and from this evaluation an F1-score of 89.24 is obtained.

The folder [code](https://github.com/bennigeir/NER/tree/main/code) contains all code, utils and configs for the project.
 - `10_fold_eval.py` : 10-fold cross-validation of the model produced
 - `SentenceGetter.py` : Parse dataset into sentences
 - `corpus_merge.py` : Old script to merge MIM-GOLD and MIM-GOLD-NER
 - `ner_api.py` : Flask API for a trained model
 - `sandbox_bert.py` : The main code; data preperation, training and evaluation (70/30)
 - `sandbox_bert_hrafn.py` : Same as `sandbox_bert.py` but with predefined train and test datasets
 - `test_ner.py` : Test a query of you choosing on the trained model

The folder [files](https://github.com/bennigeir/NER/tree/main/files) contains nothing important, only files associated with the [final report](https://github.com/bennigeir/NER/blob/main/files/report/NER.pdf).

---------------------

# Named Entity Recognition for Icelandic API

A fine tuned BERT multilingual model for NER for Icelandic is at service on www.ice-bert-ner.com.

## Open Endpoints

Open endpoints require no Authentication.

* NER : `GET /NER/?query=`
* Example: 
    `www.ice-bert-ner.com/NER/?query=Erna Sif er lektor við verkfræði- og tölvunarfræðideildir HR og forstöðumaður Svefnseturs sem nýlega var sett á fót með styrk frá Innviðasjóði.`

    Response:
    ```json
    "results": [
        [
            "[CLS]",
            "[CLS]"
        ],
        [
            "Erna",
            "B-Person"
        ],
        [
            "Sif",
            "I-Person"
        ],
        [
            "er",
            "O"
        ],
        [
            "lektor",
            "O"
        ],
        [
            "við",
            "O"
        ],
        [
            "verkfræði",
            "O"
        ],
        [
            "-",
            "X"
        ],
        [
            "og",
            "O"
        ],
        [
            "tölvunarfræðideildir",
            "O"
        ],
        [
            "HR",
            "B-Organization"
        ],
        [
            "og",
            "O"
        ],
        [
            "forstöðumaður",
            "O"
        ],
        [
            "Svefnseturs",
            "B-Organization"
        ],
        [
            "sem",
            "O"
        ],
        [
            "nýlega",
            "O"
        ],
        [
            "var",
            "O"
        ],
        [
            "sett",
            "O"
        ],
        [
            "á",
            "O"
        ],
        [
            "fót",
            "O"
        ],
        [
            "með",
            "O"
        ],
        [
            "styrk",
            "O"
        ],
        [
            "frá",
            "O"
        ],
        [
            "Innviðasjóði",
            "B-Organization"
        ],
        [
            ".",
            "O"
        ],
        [
            "[SEP]",
            "[SEP]"
        ]
    ]
