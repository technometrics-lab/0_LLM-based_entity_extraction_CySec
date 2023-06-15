# LLM-based Keyword Extraction Does not Apply Well to Cyber-Security

## Installation

To install the dependencies when cuda is available
    python3 -m pip install -r requirements_cuda.txt --extra-index-url https://pypi.nvidia.com

Otherwise
    python3 -m pip install -r requirements.txt

## Usage

To extract the keywords

    python3 pipeline_eeke.py

To draw the manifolds with the different embeddings

    python3 manifold.py

To plot the cross correlation of the models

    python3 cross_correlation_matrix_model.py

## Remarks

- When using the scripts on large data, you can have some issue with the scripts. For example: Take too long, use too much memory, ...
- There are constant on top of the files to edit some parameters of the scripts.

## Requirements

- Python3.10

## Model name

- Yake (KPE): https://pypi.org/project/yake/
- KeyBERT (KPE): https://pypi.org/project/keybert/p
- Electra-base conll03 (NER): https://huggingface.co/bhadresh-savani/electra-base-discriminator-finetuned-conll03-english
- XLM-RoBERTa-base OntoNotes5 (NER + NUM): https://huggingface.co/asahi417/tner-xlm-roberta-base-ontonotes5
- BERT COCA-docusco (TokC): https://huggingface.co/browndw/docusco-bert
- BERT-large-cased conll03 (NER): https://huggingface.co/dslim/bert-large-NER
- DistilBERT-base-uncased conll03 (NER): https://huggingface.co/elastic/distilbert-base-uncased-finetuned-conll03-english
- RoBERTa-large conll03 (NER): https://huggingface.co/Jean-Baptiste/roberta-large-ner-english
- BERT-large-uncased conll03 (NER): https://huggingface.co/Jorgeutd/bert-large-uncased-finetuned-ner
- KBIR inspec (KPE): https://huggingface.co/ml6team/keyphrase-extraction-kbir-inspec
- KBIR kpcrowd (KPE): https://huggingface.co/ml6team/keyphrase-extraction-kbir-kpcrowd
- XLM-RoBERTa-large conll03 (NER): https://huggingface.co/xlm-roberta-large-finetuned-conll03-english
- BERT-base-uncased (NER + CON R): https://huggingface.co/yanekyuk/bert-uncased-keyword-discriminator
- BERT-base-uncased (KPE): https://huggingface.co/yanekyuk/bert-uncased-keyword-extractor
- Spacy-large OntoNotes5 (NnE): https://pypi.org/project/spacy/
- Spacy-transformer OntoNotes5 (NnE): https://pypi.org/project/spacy/hu