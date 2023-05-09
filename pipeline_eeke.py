import fitz
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import os

import extractors as ex

PDF_PATH = '/data'
# number of token to extract
NB_TOKENS = 100
# Split the text in multiple part to extract the keywords
SPLIT = True
NB_PROCESS = 8

EXTRACTOR_LIST = [
    (ex.YakeExtractor,),
    (ex.KeyBERTExtractor,),
    # Not used because it's too slow
    # ex.BERTopicExtractor,
    # ex.KeyPhraseExtractor
]

for m in ex.HuggingFaceExtractor.MODEL_NAME:
    EXTRACTOR_LIST.append((ex.HuggingFaceExtractor, m))

for m in ex.SpacyExtractor.MODEL_NAME:
    EXTRACTOR_LIST.append((ex.SpacyExtractor, m))

EXTRACTOR_LIST = [EXTRACTOR_LIST[-1]]

# Class to detect the language of a text with a LLM
class LanguageDetector:
    def __init__(self):
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.tokenizer = AutoTokenizer.from_pretrained("papluca/xlm-roberta-base-language-detection")
        self.model = AutoModelForSequenceClassification.from_pretrained("papluca/xlm-roberta-base-language-detection")
        self.classifier = pipeline('sentiment-analysis', model=self.model, tokenizer=self.tokenizer, device=device)

    # Get the language of a text
    # text (str): text to analyze
    # return (list[{'label': str, 'score': int}]): list of dict with the language and the score
    def get_language(self, text):
        # If the input is too long can return an error
        try:
            return self.classifier(text[:1300])
        except Exception:
            try:
                return self.classifier(text[:514])
            except Exception:
                return [{'label': 'err'}]

# Extract text from a pdf file
# path (str|Path): path to the pdf file
# return (str): text extracted from the pdf file
def get_text_from_pdf(path):
    pdf_file = fitz.open(path)
    txt = "".join([page.get_text() for page in pdf_file])
    pdf_file.close()
    return txt

# Get all the pdf files from a directory
# path (str): path to the directory
# return (list[Path]): list of Path to the pdf files
def get_all_pdf_from_dir(path):
    return [x for x in Path(path).rglob('*.pdf')]

# Get the subcategory of a cs arxiv paper
# text (str): text of the paper
# return (str): subcategory of the paper
# throw ValueError if the category is not found
def get_category(text):
    cat_pos = text.find('[cs.')
    if cat_pos != -1:
        category = text[cat_pos+4:cat_pos+6]
    else:
        raise ValueError('Category not found')
    return category

# Remove the header and the references from a text
# text (str): text to clean
# return (str): cleaned text
def remove_header_references(text:str):
    txt_lower = text.lower()
    abstract_pos = txt_lower.find('abstract')
    if abstract_pos != -1:
        abstract_pos += len('abstract')
    else:
        # If not foud remove fixed number of characters to remove part of the header
        abstract_pos = 100

    references_pos = txt_lower.rfind('reference')
    if references_pos == -1:
        references_pos = len(text)

    return text[abstract_pos:references_pos]

# Worker to extract keywords from a list of pdf files
# model_info (tuple): tuple with the model class and the model name is needed
# pdf_list (list[Path]): list of Path to the pdf files
def worker(model_info, pdf_list):
    # load the language detector and the keyword extractor
    language_detector = LanguageDetector()
    if len(model_info) == 2:
        model = model_info[0](model_info[1], NB_TOKENS, SPLIT)
    else:
        model = model_info[0](NB_TOKENS, SPLIT)

    i = 0
    print(f'start {str(model)}', flush=True)
    for pdf in pdf_list:
        txt = get_text_from_pdf(pdf)
        try:
            category = get_category(txt)
        except ValueError:
            category = None
        if os.path.exists(model.get_file_name(category, pdf.stem)):
            continue
        # process only english papers or papers with no language detected since it is mostly english
        if language_detector.get_language(txt)[0]['label'] not in ['en', 'err']:
            continue

        txt = remove_header_references(txt)
        keywords = model.extract_keywords(txt)
        model.save_result(keywords, category, pdf.stem)

        # show progress
        if i % 30 == 0:
            print(f'{str(model)} {i}', flush=True)
        i += 1


def main():
    print('***** INIT *****', flush=True)
    pdf_list = get_all_pdf_from_dir(PDF_PATH)

    print('***** START *****', flush=True)
    with ProcessPoolExecutor(NB_PROCESS, mp_context=multiprocessing.get_context('spawn')) as pool:
        tasks = [pool.submit(worker, extractor, pdf_list) for extractor in EXTRACTOR_LIST]
        print('***** WAITING *****', flush=True)
        [t.result() for t in tasks]

    print('***** FINISH *****', flush=True)

if __name__ == '__main__':
    main()
