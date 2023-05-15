import os
from pathlib import Path
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from typing import Union
import json

import fitz
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch

import extractors as ex

CONFIG = json.load(open("config.json", "r"))

EXTRACTOR_LIST = [
    (ex.YakeExtractor,),
    (ex.KeyBERTExtractor,),
    # Not used because it's too slow
    # ex.BERTopicExtractor,
    # ex.KeyPhraseExtractor
]

for model_name in ex.HuggingFaceExtractor.MODEL_NAMES:
    EXTRACTOR_LIST.append((ex.HuggingFaceExtractor, model_name))

for model_name in ex.SpacyExtractor.MODEL_NAMES:
    EXTRACTOR_LIST.append((ex.SpacyExtractor, model_name))

#TODO remove next line
EXTRACTOR_LIST = EXTRACTOR_LIST[3:4]

class LanguageDetector:

    """Detect the language of a text with a LLM"""

    language_model_name = "papluca/xlm-roberta-base-language-detection"

    def __init__(self):
        device = (
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.language_model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.language_model_name)
        self.classifier = pipeline(
            "sentiment-analysis",
            model=self.model,
            tokenizer=self.tokenizer,
            device=device,
        )

    def get_language(self, text: str) -> list[dict]:
        """Get the language of a text

        Parameters:
            text (int): text to analyze

        Return:
            language_score list[{'label': str, 'score': int}]: list of dict with the language and the score
        """

        # If the input is too long can return an error
        try:
            return self.classifier(text[:1300])
        except Exception:
            try:
                return self.classifier(text[:514])
            except Exception:
                return [{"label": "err"}]


def get_text_from_pdf(path: Union[str, Path]) -> str:
    """Extract text from a pdf file

    Parameters:
        path (str|Path): path to the pdf file

    Return:
        text (str): text extracted from the pdf file
    """
    pdf_file = fitz.open(path)
    txt = "".join([page.get_text() for page in pdf_file])
    pdf_file.close()
    return txt


def get_all_pdf_from_dir(path: str) -> list[Path]:
    """Get all the pdf files from a directory

    Parameters:
        path (str): path to the directory

    Return:
        pdf_list (list[Path]): list of Path to the pdf files
    """
    return [x for x in Path(path).rglob("*.pdf")]


def get_category(text: str) -> str:
    """Get the category of a cs arxiv paper

    Parameters:
        text (str): text of the paper

    Return:
        category (str): category of the paper

    Throw:
        ValueError: if the category is not found
    """
    cat_pos = text.find("[cs.")
    if cat_pos != -1:
        # to keep only the subcategory
        category = text[cat_pos + 4 : cat_pos + 6]
    else:
        raise ValueError("Category not found")
    return category


def remove_header_references(text: str) -> str:
    """Remove the header and the references from a text

    Parameters:
        text (str): text to clean

    Return:
        text (str): cleaned text
    """
    txt_lower = text.lower()
    abstract_pos = txt_lower.find("abstract")
    if abstract_pos != -1:
        abstract_pos += len("abstract")
    else:
        # If not foud remove fixed number of characters to remove part of the header
        abstract_pos = 100

    references_pos = txt_lower.rfind("reference")
    if references_pos == -1:
        references_pos = len(text)

    return text[abstract_pos:references_pos]


def worker(model_info: tuple, pdf_list: list[Path]):
    """Worker to extract keywords from a list of pdf files

    Parameters:
        model_info (tuple): tuple with the model class and the model name is needed
        pdf_list (list[Path]): list of Path to the pdf files
    """
    # load the language detector and the keyword extractor
    language_detector = LanguageDetector()
    if len(model_info) == 2:
        model = model_info[0](model_info[1], CONFIG["PICKLE_PATH"], CONFIG["NB_TOKENS"], CONFIG["SPLIT"])
    else:
        model = model_info[0](CONFIG["PICKLE_PATH"], CONFIG["NB_TOKENS"], CONFIG["SPLIT"])

    i = 0
    print(f"start {str(model)}", flush=True)
    for pdf in pdf_list:
        txt = get_text_from_pdf(pdf)
        try:
            category = get_category(txt)
        except ValueError:
            category = None
        if os.path.exists(model.get_file_name(category, pdf.stem)):
            continue
        # process only english papers or papers with no language detected since it is mostly english
        if language_detector.get_language(txt)[0]["label"] not in ["en", "err"]:
            continue

        txt = remove_header_references(txt)
        keywords = model.extract_keywords(txt)
        model.save_result(keywords, category, pdf.stem)

        # show progress
        if i % 30 == 0:
            print(f"{str(model)} {i}", flush=True)
        i += 1


def main():
    """Process all the pdf files in the directory and extract the keywords with the different extractors"""

    print("***** INIT *****", flush=True)
    pdf_list = get_all_pdf_from_dir(CONFIG["PDF_PATH"])

    print("***** START *****", flush=True)
    with ProcessPoolExecutor(
        CONFIG["NB_PROCESS"], mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        tasks = [
            pool.submit(worker, extractor, pdf_list) for extractor in EXTRACTOR_LIST
        ]
        print("***** WAITING *****", flush=True)
        [t.result() for t in tasks]

    print("***** FINISH *****", flush=True)


if __name__ == "__main__":
    main()
