import os
from pathlib import Path
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from typing import Union
import json
from itertools import product
import time

import fitz
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch

import extractors as ex
from flowcontrol.registration import SavePaths


CONFIG = json.load(open("src/flowcontrol/config.json", "r"))

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


class LanguageDetector:

    """Detect the language of a text with a LLM"""

    language_model_name = "papluca/xlm-roberta-base-language-detection"

    def __init__(self):
        device = (
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.language_model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.language_model_name
        )
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
    return list(Path(path).rglob("*.pdf"))


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
    introduction_pos = txt_lower.find("introduction")

    if introduction_pos != -1 and abstract_pos != -1:
        abstract_pos = min(abstract_pos, introduction_pos)
    else:
        abstract_pos = max(abstract_pos, introduction_pos)

    if abstract_pos == -1:
        # If not foud remove fixed number of characters to remove part of the header
        abstract_pos = 100

    references_pos = txt_lower.rfind("reference")
    acknowledgements_pos = txt_lower.rfind("acknowledgement")
    if (
        acknowledgements_pos != -1
        and acknowledgements_pos < references_pos
        and acknowledgements_pos > len(text) / 2
    ):
        references_pos = acknowledgements_pos
    if references_pos == -1:
        references_pos = len(text)

    return text[abstract_pos:references_pos]


def retry_on_error(func, error_to_catch, max_attempts=3, *args):
    """Retry a function if an error is raised

    Parameters:
        func (function): function to retry
        error_to_catch (Exception): error to catch
        max_attempts (int): number of attempts
    """
    for attempt in range(max_attempts):
        try:
            ret = func(*args)
            break
        except error_to_catch as e:
            if attempt >= max_attempts - 1:
                raise e
            time.sleep(10)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return ret


def worker(active_savepath, model_info: tuple, pdf_list: list[Path]):
    """Worker to extract keywords from a list of pdf files

    Parameters:
        active_savepath (SavePaths): SavePaths object to save the results
        model_info (tuple): tuple with the model class and the model name is needed
        pdf_list (list[Path]): list of Path to the pdf files
    """
    # load the language detector and the keyword extractor
    language_detector = LanguageDetector()
    if len(model_info) == 2:
        model = model_info[0](model_info[1], CONFIG["NB_TOKENS"], CONFIG["SPLIT"])
    else:
        model = model_info[0](CONFIG["NB_TOKENS"], CONFIG["SPLIT"])

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
        keywords = retry_on_error(model.extract_keywords, RuntimeError, 3, txt)

        active_savepath.logging(
            f"keywords of model {model}: {len(keywords)} {keywords[:10]}"
        )
        model.save_result(keywords, active_savepath.result, category, pdf.stem)

        # show progress
        if i % 30 == 0:
            print(f"{str(model)} {i}", flush=True)
        i += 1

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    """Process all the pdf files in the directory and extract the keywords with the different extractors"""

    active_savepath = SavePaths("pipeline_eeke")
    print("***** INIT *****", flush=True)
    pdf_list = get_all_pdf_from_dir(CONFIG["PDF_PATH"])
    active_savepath.logging(f"pdf_list: {len(pdf_list)} {pdf_list[:10]}")
    active_savepath.logging(f"extractor list: {len(EXTRACTOR_LIST)} {EXTRACTOR_LIST}")
    active_savepath.logging("Start processing")

    print("***** START *****", flush=True)

    chunk_size = 500
    pdf_list_chunked = [
        pdf_list[x : x + chunk_size] for x in range(0, len(pdf_list), chunk_size)
    ]
    with ProcessPoolExecutor(
        CONFIG["NB_PROCESS"],
        mp_context=multiprocessing.get_context("spawn"),
        max_tasks_per_child=1,
    ) as pool:
        tasks = [
            pool.submit(worker, active_savepath, extractor, pdfs)
            for pdfs, extractor in product(pdf_list_chunked, EXTRACTOR_LIST)
        ]
        print("***** WAITING *****", flush=True)
        [t.result() for t in tasks]

    print("***** FINISH *****", flush=True)
    active_savepath.logging("Finish processing")
    active_savepath.move_result_to_final_path()


if __name__ == "__main__":
    main()
