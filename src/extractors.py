import warnings
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning

warnings.simplefilter("ignore", category=NumbaDeprecationWarning)

import os
import re
import pickle
from collections import Counter
from abc import ABC, abstractmethod
import warnings

import spacy
from spacy.matcher import Matcher
import torch
from lsg_converter import LSGConverter
from keybert import KeyBERT
import yake

# from bertopic import BERTopic
from keyphrasetransformer import KeyPhraseTransformer
from sklearn.feature_extraction.text import CountVectorizer
from keyphrase_vectorizers import KeyphraseCountVectorizer
from bertopic.vectorizers import ClassTfidfTransformer
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline


class Extractor(ABC):

    """Abstract class for extractors"""

    def __init__(self, name, model, nb_tokens, splitlines):
        """Init the extractor

        Parameters:
            name (str): name of the extractor
            model (object): model used to extract keywords
            pickle_path (str): path where to save the result
            nb_tokens (int): number of keywords to extract
            splitlines (bool): if True, split the text into lines before extracting keywords
                               currently only used by HuggingFaceExtractor
        """
        self.name = name
        self.model = model
        self.nb_tokens = nb_tokens
        self.splitlines = splitlines

    def get_file_name(self, category=None, pdf_name=""):
        """Return file name to save the result

        Parameters:
            category (str) default=None: category of the pdf
            pdf_name (str) default='': name of the pdf

        Return:
            file_name (str): file name where to save the result
        """
        if category is None:
            category = "none/"
        else:
            category = f"{category}/"
        if pdf_name != "":
            pdf_name = pdf_name + "/"
        split_str = "_split" if self.splitlines else ""
        return f"pickle_model_keyword/{category}{pdf_name}{self.name}_{self.nb_tokens}{split_str}.pkl"

    @abstractmethod
    def extract_keywords(self, doc: str) -> list[str]:
        """Extract keywords from a document

        Parameters:
            doc (str): a text document

        Return:
            keywords (list[str]): list of keywords"""
        pass

    def save_result(
        self, keywords, root_path, category=None, pdf_name="", remove_score=False
    ):
        """Save the extracted keywords

        Parameters:
            keywords (list[(str, float)]): list of keywords with their score
            root_path (str): path where to save the result
            category (str) default=None: category of the pdf
            pdf_name (str) default='': name of the pdf
            remove_score (bool) default=True: if True, remove the score from the keywords
        """
        if remove_score:
            keywords = [word[0] for word in keywords]

        file_name = f"{root_path}/{self.get_file_name(category, pdf_name)}"
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        with open(file_name, "wb") as f:
            pickle.dump(keywords, f)

    def check_parenthesis(self, txt: str) -> bool:
        """Check if parenthesis are balanced

        Parameters:
            txt (str): text to check

        Returns:
            bool: True if parenthesis are balanced, False otherwise
        """

        open_list = ["[", "{", "("]
        close_list = ["]", "}", ")"]
        stack = []
        for i in txt:
            if i in open_list:
                stack.append(i)
            elif i in close_list:
                pos = close_list.index(i)
                if stack and (open_list[pos] == stack[-1]):
                    stack.pop()
                else:
                    return False
        if len(stack) == 0:
            return True
        return False

    def clear_filter_keywords(self, keywords: list[str]):
        """Clear and filter keywords

        Parameters:
            keywords (list[str]): list of keywords

        Returns:
            list[str]: list of filtered keywords
        """
        # remove non ascii tokens and with parenthesis not balanced
        keywords = [
            keyword.strip()
            for keyword in keywords
            if keyword.isascii() and self.check_parenthesis(keyword)
        ]

        for i in range(len(keywords)):
            keyword = keywords[i]
            keyword = keyword.replace("\n", " ")  # remove new lines
            keyword = re.sub(" \d* ", " ", keyword)  # remove numbers
            keyword = re.sub(
                "(\s*-\s+|\s+-\s*)", "", keyword
            )  # remove hyphens surronunded by spaces

            # if all letters of a word are upper case lower it
            split_keyword = keyword.split(" ")
            for j in range(len(split_keyword)):
                if split_keyword[j].isupper():
                    split_keyword[j] = split_keyword[j].lower()
                if len(split_keyword[j]) <= 2:
                    split_keyword[j] = ""
            keyword = " ".join([word for word in split_keyword if word != ""])

            # split word by capital letters if there are no spaces
            keyword = " ".join(
                [
                    catch_group[0]
                    for catch_group in re.findall(
                        "(([\da-z]+|[A-Z\.]+)[^A-Z\s()\[\]{}]*)", keyword
                    )
                ]
            )
            keyword = keyword.lower()  # put to lower case
            keyword = re.sub("\s{2,}", " ", keyword)  # remove double spaces
            keyword = re.sub("[,\.']", "", keyword)  # remove punctuation

            if len(keyword) > 0 and keyword[-1] == "-":
                keyword = keyword[:-1]
            keywords[i] = keyword.strip()  # remove leading and trailing spaces

        return [keyword for keyword in keywords if len(keyword) > 2]

    def __str__(self):
        return self.name


class YakeExtractor(Extractor):
    """Yake extractor extends Extractor class"""

    def __init__(self, nb_tokens, splitlines):
        """Init the extractor

        Parameters:
            nb_tokens (int): number of keywords to extract
            splitlines (bool): if True, split the text into lines before extracting keywords
        """
        super().__init__("yake", yake.KeywordExtractor(), nb_tokens, splitlines)

    def extract_keywords(self, doc):
        keywords = self.model.extract_keywords(doc)
        keywords = sorted(keywords, key=lambda x: x[1], reverse=True)
        keywords = [x[0] for x in keywords]
        keywords = self.clear_filter_keywords(keywords)
        return keywords[: self.nb_tokens]


# class BERTopicExtractor(Extractor):
#     """BERTopic extractor extends Extractor class"""
#     def __init__(self, pickle_path, nb_tokens, splitlines):
#         """Init the extractor
#
#         Parameters:
#             pickle_path (str): path where to save the result
#             nb_tokens (int): number of keywords to extract
#             splitlines (bool): if True, split the text into lines before extracting keywords
#         """
#         vectorizer = te_utils.get_vectorizer()
#         if torch.cuda.is_available():
#             umap_model = UMAP(n_components=5, n_neighbors=15, min_dist=0.0)
#             hdbscan_model = HDBSCAN(min_samples=10, gen_min_span_tree=True, prediction_data=True)
#             topic_model = BERTopic(umap_model=umap_model, hdbscan_model=hdbscan_model, min_topic_size=10, vectorizer_model=vectorizer, nr_topics='auto')
#         else:
#             topic_model = BERTopic(min_topic_size=10, vectorizer_model=vectorizer, nr_topics='auto')
#
#
#         super().__init__('bertopic', pickle_path, topic_model, nb_tokens, splitlines)
#
#     def extract_keywords(self, docs):
#         words = docs.split(' ')
#         docs = [' '.join(words[i:i+100]) for i in range(0, len(words), 100)]
#
#         topics, probs = self.model.fit_transform(docs)
#         doc_info = self.model.get_document_info(docs)
#         top_label = self.model.generate_topic_labels(topic_prefix=False, separator=', ')
#
#         topic_doc = list(set(doc_info['Topic']))[:self.nb_tokens]
#
#         keywords = [(top_label[y], np.nan) for y in topic_doc]
#         return keywords


class KeyBERTExtractor(Extractor):
    """KeyBERT extractor extends Extractor class"""

    SPACY_REMOVAL = [
        "ADV",
        "AUX",
        "CONJ",
        "INTJ",
        "PRON",
        "PROPN",
        "CCONJ",
        "PUNCT",
        "PART",
        "SCONJ",
        "VERB",
        "DET",
        "ADP",
        "SPACE",
        "NUM",
        "SYM",
    ]

    def __init__(
        self,
        nb_tokens,
        splitlines,
        model="keyphrase",
        spacy_removal=SPACY_REMOVAL,
    ):
        """Init the extractor

        Parameters:
            nb_tokens (int): number of keywords to extract
            splitlines (bool): if True, split the text into lines before extracting keywords
            model (str) default='keyphrase': model to use
                possible values: 'keyphrase', 'tfidf', 'count'
            spacy_removal (list[str]) default=SPACY_REMOVAL: list of spacy removal
        """
        self.vectorizer = self._get_vectorizer(model, spacy_removal)
        super().__init__("keybert", KeyBERT(), nb_tokens, splitlines)

    def _get_vectorizer(self, model="keyphrase", spacy_removal=SPACY_REMOVAL):
        """Return the vectorizer to use

        Parameters:
            model (str) default='keyphrase': model to use
                possible values: 'keyphrase', 'tfidf', 'count'
            spacy_removal (list[str]) default=SPACY_REMOVAL: list of spacy removal

        Return:
            vectorizer (function): vectorizer to use
        """
        match model:
            case "keyphrase":
                return KeyphraseCountVectorizer(
                    spacy_exclude=spacy_removal, spacy_pipeline="en_core_web_lg"
                )
            case "tfidf":
                return ClassTfidfTransformer(reduce_frequent_words=True)
            case "count":
                CountVectorizer(stop_words="english")
            case _:
                raise ValueError(f"Unknown vectorizer model: {model}")

    def extract_keywords(self, doc):
        try:
            keywords = self.model.extract_keywords(
                doc,
                use_mmr=True,
                diversity=0.7,
                vectorizer=self.vectorizer,
                top_n=int(self.nb_tokens * 1.3),
            )
            keywords = [x[0] for x in keywords]
            keywords = self.clear_filter_keywords(keywords)
            return keywords[: self.nb_tokens]
        except Exception:
            return []


class KeyPhraseExtractor(Extractor):
    """KeyPhrase extractor extends Extractor class"""

    def __init__(self, nb_tokens, splitlines):
        """Init the extractor

        Parameters:
            nb_tokens (int): number of keywords to extract
            splitlines (bool): if True, split the text into lines before extracting keywords
        """
        super().__init__("keyphrase", KeyPhraseTransformer(), nb_tokens, splitlines)

    def extract_keywords(self, doc):
        keywords = self.model.get_key_phrases(doc)
        keywords = self.clear_filter_keywords(keywords)

        return keywords[: self.nb_tokens]


class HuggingFaceExtractor(Extractor):
    """HuggingFace extractor extends Extractor class"""

    # list of models that can be used
    MODEL_NAMES = [
        "asahi417/tner-xlm-roberta-base-ontonotes5",
        "bhadresh-savani/electra-base-discriminator-finetuned-conll03-english",
        "browndw/docusco-bert",  # mismatched size, solved with lsg_converter
        "dslim/bert-large-NER",
        "elastic/distilbert-base-uncased-finetuned-conll03-english",
        "Jean-Baptiste/roberta-large-ner-english",
        "Jorgeutd/bert-large-uncased-finetuned-ner",
        # 'jplu/tf-xlm-r-ner-40-lang', #tf problem
        "ml6team/keyphrase-extraction-kbir-inspec",
        "ml6team/keyphrase-extraction-kbir-kpcrowd",
        # 'ml6team/keyphrase-extraction-kbir-kptimes', #mismatched size
        # 'ml6team/keyphrase-extraction-kbir-openkp', #mismatched size
        # 'ml6team/keyphrase-extraction-kbir-semeval2017', #mismatched size
        "xlm-roberta-large-finetuned-conll03-english",
        "yanekyuk/bert-uncased-keyword-discriminator",
        "yanekyuk/bert-uncased-keyword-extractor",
    ]

    # is_lsg (bool) default=False: if True, use the lsg_converter to convert the model to a lsg model
    def __init__(self, model_name, nb_tokens, splitlines, is_lsg=False):
        """Init the extractor

        Parameters:
            model_name (str): name of the model to use. See MODEL_NAMES for available models
            nb_tokens (int): number of keywords to extract
            splitlines (bool): if True, split the text into lines before extracting keywords
            is_lsg (bool) default=False: if True, use the lsg_converter to convert the model to a lsg model
        """

        # get model from huggingface and load it on GPU if available
        device = (
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForTokenClassification.from_pretrained(model_name)
        classifier = pipeline(
            "ner",
            model=model,
            tokenizer=tokenizer,
            aggregation_strategy="max",
            device=device,
        )

        # convert to lsg model, this method allow to send larger input to the model, this implies a downgrade in performance
        converter = LSGConverter(max_sequence_length=8192)
        arch_list = ["Bert", "DistilBert", "Roberta", "Electra", "XLMRoberta"]
        self.lsg_model = None
        for a in arch_list:
            try:
                lsg_model, lsg_tokenizer = converter.convert_from_pretrained(
                    model_name, architecture=f"{a}ForTokenClassification"
                )
                self.lsg_model = pipeline(
                    "ner",
                    model=lsg_model,
                    tokenizer=lsg_tokenizer,
                    aggregation_strategy="max",
                    device=device,
                )
                break
            except Exception:
                # print(e)
                continue

        super().__init__(model_name, classifier, nb_tokens, splitlines)
        self.is_lsg = is_lsg
        self.tokenizer = tokenizer

    def get_file_name(self, category=None, pdf_name=""):
        if category is None:
            category = "none/"
        else:
            category = f"{category}/"
        if pdf_name != "":
            pdf_name = pdf_name + "/"
        split_str = "_split" if self.splitlines else ""
        lsg_str = "_lsg" if self.is_lsg else ""
        return f'pickle_model_keyword/{category}{pdf_name}hugging_{self.name.replace("/", "_")}_{self.nb_tokens}{split_str}{lsg_str}.pkl'

    def _get_split_indexes(self, doc: str):
        """Split the document in smaller chunks to avoid memory issues if splitlines is True

        Parameters:
            doc (str): document to split

        Returns:
            list: list of indexes to split the document
        """
        txt_len = len(doc.split(" "))
        if self.splitlines:
            token_len = len(self.tokenizer(doc)["input_ids"])
            tok_tex_factor = token_len / txt_len

            split_index = list(
                range(
                    0,
                    token_len,
                    int(min(self.tokenizer.max_model_input_sizes.values()) * 0.95),
                )
            )
            split_index = [int(x / tok_tex_factor) for x in split_index] + [txt_len]
        else:
            split_index = [0, txt_len]

        return split_index

    def extract_keywords(self, doc):
        results = []

        if doc is not None and doc.strip() != "":
            split_index = self._get_split_indexes(doc)

            for start, end in zip(split_index[:-1], split_index[1:]):
                doc_split = " ".join(doc.split(" ")[start:end])

                try:
                    if self.is_lsg and self.lsg_model is not None:
                        results.extend(self.lsg_model(doc_split))
                    else:
                        results.extend(self.model(doc_split))
                except Exception:
                    if self.lsg_model is not None:
                        try:
                            results.extend(self.lsg_model(doc_split))
                        except Exception:
                            continue

        keyword = list(
            set(
                [
                    word["word"]
                    for word in sorted(results, key=lambda x: x["score"], reverse=True)
                    if len(word["word"]) > 1
                ]
            )
        )
        keyword = self.clear_filter_keywords(keyword)
        return keyword[: self.nb_tokens]


class SpacyExtractor(Extractor):
    """Spacy extractor extends Extractor class"""

    MODEL_NAMES = [
        "en_core_web_lg",
        "en_core_web_trf",
    ]

    RULES = {
        "Noun and specific dep": [
            {
                "DEP": {"IN": ["compound", "amod", "ccomp", "conj", "nmod"]},
                "OP": "{1,}",
            },
            {"POS": {"IN": ["NOUN", "X", "PROPN"]}},
        ],
        "Noun and adjective": [{"POS": {"IN": ["NOUN", "X", "PROPN"]}}, {"POS": "ADJ"}],
        "Noun": [
            {
                "POS": {"IN": ["NOUN", "X", "PROPN"]},
                "LIKE_EMAIL": False,
                "LIKE_URL": False,
            }
        ],
        # "Noun and verb": [{"POS": {"IN": ["NOUN", "X", "PROPN", "ADJ"]}}, {"POS": "VERB"}],
        # "Noun and amod": [{"DEP": "vocative", "OP": "{1,}"}, {"POS": {"IN": ["NOUN", "X", "PROPN"]}}],
    }

    CHUNK_SIZE = 990_000

    def __init__(self, model_name, nb_tokens, splitlines):
        """Init the extractor

        Parameters:
            model_name (str): name of the model to use. See MODEL_NAMES for available models
            nb_tokens (int): number of keywords to extract
            splitlines (bool): if True, split the text into lines before extracting keywords
        """
        spacy.prefer_gpu()
        nlp = spacy.load(model_name)
        nlp.add_pipe("doc_cleaner")
        # nlp.add_pipe("merge_entities")

        self.rule_matcher = Matcher(nlp.vocab)
        for rule_name, rule_tags in self.RULES.items():  # register rules in matcher
            self.rule_matcher.add(rule_name, [rule_tags])

        super().__init__(model_name, nlp, nb_tokens, splitlines)

    def _remove_sub_interval(self, matches: list[tuple[int, int, int]]):
        """Remove sub intervals from a list of intervals

        Parameters:
            matches (list[tuple[int, int, int]]): list of intervals (match_id, start, end)

        Returns:
            list[tuple[int, int, int]]: list of intervals without sub intervals
        """
        keep_matches = []

        if len(matches) > 0:
            _, longest_start_id, longest_end_id = matches[0]
            id_to_add = 0
            for i, match in enumerate(matches[1:], 1):
                _, start_id, end_id = match
                if (longest_start_id <= end_id and longest_end_id >= end_id) or (
                    start_id <= longest_end_id and end_id >= longest_start_id
                ):
                    if end_id - start_id > longest_end_id - longest_start_id:
                        longest_start_id, longest_end_id = start_id, end_id
                        id_to_add = i
                else:
                    if end_id > longest_end_id:
                        keep_matches.append(matches[id_to_add])
                        longest_start_id, longest_end_id = start_id, end_id
                        id_to_add = i
        return keep_matches

    def extract_keywords(self, doc):
        results = []

        # split doc into chunks
        chunks_file = []
        curr_return_line = 0
        next_return_line = doc.find("\n", curr_return_line + self.CHUNK_SIZE) + 1
        while next_return_line > curr_return_line:
            chunks_file.append(doc[curr_return_line:next_return_line])
            curr_return_line = next_return_line
            next_return_line = doc.find("\n", curr_return_line + self.CHUNK_SIZE) + 1
        chunks_file.append(doc[curr_return_line:])

        for chunk in chunks_file:
            tokens = self.model(chunk)
            matches = self.rule_matcher(tokens)
            keep_matches = self._remove_sub_interval(matches)
            results.extend(tokens[start:end].lemma_ for _, start, end in keep_matches)

        # search for the most common tokens
        if len(results) > 0:
            keywords = self.clear_filter_keywords(results)
            return [
                word for word, count in Counter(keywords).most_common() if count > 3
            ][: self.nb_tokens]
        return []
