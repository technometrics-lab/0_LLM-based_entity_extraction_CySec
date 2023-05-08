import yake
#from bertopic import BERTopic
from keybert import KeyBERT
from keyphrasetransformer import KeyPhraseTransformer

from keyphrase_vectorizers import KeyphraseCountVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.vectorizers import ClassTfidfTransformer

from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from lsg_converter import LSGConverter
import spacy

import pickle
import numpy as np
import torch
from collections import Counter
import os

from abc import ABC, abstractmethod

# abstract class Extractor(ABC):
class Extractor(ABC):
    PIKLE_PATH = 'results/pickle_model_keyword'

    # name (str): name of the extractor
    # model (object): model used to extract keywords
    # nb_tokens (int): number of keywords to extract
    # splitlines (bool): if True, split the text into lines before extracting keywords
    #                    currently only used by HuggingFaceExtractor
    def __init__(self, name, model, nb_tokens, splitlines):
        self.name = name
        self.model = model
        self.nb_tokens = nb_tokens
        self.splitlines = splitlines

    # Return file name to save the result
    # category (str) default=None: category of the pdf
    # pdf_name (str) default='': name of the pdf
    def get_file_name(self, category=None, pdf_name=''):
        if category is None:
            category = 'none/'
        else:
            category = f'{category}/'
        if pdf_name != '':
            pdf_name = pdf_name + '/'
        split_str = '_split' if self.splitlines else ''
        return f'{self.PIKLE_PATH}/{category}{pdf_name}{self.name}_{self.nb_tokens}{split_str}.pkl'

    # Extract keywords from a document
    # doc (str): a text document
    @abstractmethod
    def extract_keywords(self, doc):
        pass

    # save the extracted keywords
    # keywords (list[(str, float)]): list of keywords with their score
    # category (str) default=None: category of the pdf
    # pdf_name (str) default='': name of the pdf
    # remove_score (bool) default=True: if True, remove the score from the keywords
    def save_result(self, keywords, category=None, pdf_name='', remove_score=True):
        if remove_score:
            keywords = [x[0] for x in keywords]

        file_name = self.get_file_name(category, pdf_name)
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        with open(file_name, 'wb') as f:
            pickle.dump(keywords, f)

    def __str__(self):
        return self.name

# yake
class YakeExtractor(Extractor):
    def __init__(self, nb_tokens, splitlines):
        super().__init__('yake', yake.KeywordExtractor(), nb_tokens, splitlines)

    def extract_keywords(self, docs):
        keywords = self.model.extract_keywords(docs)[:-self.nb_tokens -1:-1]
        return keywords


#BERToppic
# class BERTopicExtractor(Extractor):
#     def __init__(self, nb_tokens, splitlines):
#         vectorizer = te_utils.get_vectorizer()
#         if torch.cuda.is_available():
#             umap_model = UMAP(n_components=5, n_neighbors=15, min_dist=0.0)
#             hdbscan_model = HDBSCAN(min_samples=10, gen_min_span_tree=True, prediction_data=True)
#             topic_model = BERTopic(umap_model=umap_model, hdbscan_model=hdbscan_model, min_topic_size=10, vectorizer_model=vectorizer, nr_topics='auto')
#         else:
#             topic_model = BERTopic(min_topic_size=10, vectorizer_model=vectorizer, nr_topics='auto')
# 
# 
#         super().__init__('bertopic', topic_model, nb_tokens, splitlines)
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


#KeyBERT
class KeyBERTExtractor(Extractor):
    SPACY_REMOVAL = ['ADV', 'AUX', 'CONJ', 'INTJ','PRON','PROPN', 'CCONJ','PUNCT','PART', 'SCONJ', 'VERB','DET','ADP','SPACE', 'NUM', 'SYM']

    def __init__(self, nb_tokens, splitlines, model='keyphrase', spacy_removal=SPACY_REMOVAL):
        self.vectorizer = self._get_vectorizer(model, spacy_removal)
        super().__init__('keybert', KeyBERT(), nb_tokens, splitlines)

    def _get_vectorizer(self, model='keyphrase', spacy_removal=SPACY_REMOVAL):
        match model:
            case 'keyphrase':
                return KeyphraseCountVectorizer(spacy_exclude=spacy_removal, spacy_pipeline='en_core_web_lg')
            case 'tfidf':
                return ClassTfidfTransformer(reduce_frequent_words=True)
            case 'count':
                CountVectorizer(stop_words='english')
            case _:
                raise ValueError(f'Unknown vectorizer model: {model}')

    def extract_keywords(self, docs):
        try:
            keywords = self.model.extract_keywords(docs, use_mmr=True, diversity=0.7, vectorizer=self.vectorizer, top_n=self.nb_tokens)
            return keywords
        except:
            return []


#KeyPhraseTransformer
class KeyPhraseExtractor(Extractor):
    def __init__(self, nb_tokens, splitlines):
        super().__init__('keyphrase', KeyPhraseTransformer(), nb_tokens, splitlines)

    def extract_keywords(self, docs):
        keywords = self.model.get_key_phrases(docs)[:self.nb_tokens]

        keywords = [[(y, np.nan) for y in x] for x in keywords]

        return keywords


#HuggingFace
class HuggingFaceExtractor(Extractor):
    #list of models that can be used
    MODEL_NAME = [
        'asahi417/tner-xlm-roberta-base-ontonotes5',
        'bhadresh-savani/electra-base-discriminator-finetuned-conll03-english',
        'browndw/docusco-bert', #mismatched size, solved with lsg_converter
        'dslim/bert-large-NER',
        'elastic/distilbert-base-uncased-finetuned-conll03-english',
        'Jean-Baptiste/roberta-large-ner-english',
        'Jorgeutd/bert-large-uncased-finetuned-ner',
        # 'jplu/tf-xlm-r-ner-40-lang', #tf problem
        'ml6team/keyphrase-extraction-kbir-inspec',
        'ml6team/keyphrase-extraction-kbir-kpcrowd',
        # 'ml6team/keyphrase-extraction-kbir-kptimes', #mismatched size
        # 'ml6team/keyphrase-extraction-kbir-openkp', #mismatched size
        # 'ml6team/keyphrase-extraction-kbir-semeval2017', #mismatched size
        'xlm-roberta-large-finetuned-conll03-english',
        'yanekyuk/bert-uncased-keyword-discriminator',
        'yanekyuk/bert-uncased-keyword-extractor',
    ]

    # is_lsg (bool) default=False: if True, use the lsg_converter to convert the model to a lsg model
    def __init__(self, model_name, nb_tokens, splitlines, is_lsg=False):
        #get model from huggingface and load it on GPU if available
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForTokenClassification.from_pretrained(model_name)
        classifier = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy='max', device=device)

        #convert to lsg model, this method allow to send larger input to the model, this implies a downgrade in performance
        converter = LSGConverter(max_sequence_length=8192)
        arch_list = ['Bert', 'DistilBert', 'Roberta', 'Electra', 'XLMRoberta']
        self.lsg_model = None
        for a in arch_list:
            try:
                lsg_model, lsg_tokenizer = converter.convert_from_pretrained(model_name, architecture=f'{a}ForTokenClassification')
                self.lsg_model = pipeline("ner", model=lsg_model, tokenizer=lsg_tokenizer, device_map='auto', aggregation_strategy='max', device=device)
                break
            except Exception as e:
                # print(e)
                continue

        super().__init__(model_name, classifier, nb_tokens, splitlines)
        self.is_lsg = is_lsg
        self.tokenizer = tokenizer

    def get_file_name(self, category=None, pdf_name=''):
        if category is None:
            category = 'none/'
        else:
            category = f'{category}/'
        if pdf_name != '':
            pdf_name = pdf_name + '/'
        split_str = '_split' if self.splitlines else ''
        lsg_str = '_lsg' if self.is_lsg else ''
        return f'{self.PIKLE_PATH}/{category}{pdf_name}hugging_{self.name.replace("/", "_")}_{self.nb_tokens}{split_str}{lsg_str}.pkl'


    def extract_keywords(self, docs):
        results = []

        if docs is not None and docs.strip() !='':
            txt_len = len(docs.split(' '))
            if self.splitlines:
                token_len = len(self.tokenizer(docs)['input_ids'])
                tok_tex_factor = token_len / txt_len

                split_index = list(range(0, token_len, int(min(self.tokenizer.max_model_input_sizes.values())*0.95)))
                split_index = [int(x / tok_tex_factor) for x in split_index] + [txt_len]
            else:
                split_index = [0, txt_len]

            for s, e in zip(split_index[:-1], split_index[1:]):
                doc_split = ' '.join(docs.split(' ')[s:e])
                
                try:
                    if self.is_lsg and self.lsg_model is not None:
                        results.extend(self.lsg_model(doc_split))
                    else:
                        results.extend(self.model(doc_split))
                except Exception as e:
                    if self.lsg_model is not None:
                        try:
                            results.extend(self.lsg_model(doc_split))
                        except:
                            continue

        return list(set([x['word'] for x in sorted(results, key=lambda x: x['score'], reverse=True) if len(x['word']) > 1]))[:self.nb_tokens]


#Spacy
class SpacyExtractor(Extractor):
    MODEL_NAME = [
        'en_core_web_lg',
        'en_core_web_trf',
    ]

    def __init__(self, model_name, nb_tokens, splitlines):
        spacy.prefer_gpu()
        nlp = spacy.load(model_name)
        nlp.add_pipe('merge_entities')

        super().__init__(model_name, nlp, nb_tokens, splitlines)

    def extract_keywords(self, docs):
        results = []
        if docs is not None:# and docs.strip() !='':
            if len(docs) > 1000:
                for i in range(0, len(docs), 1000):
                    results.extend(self.model(docs[i:i+1000].lower()))
            else:
                results = self.model(docs.lower())

        # search for the most common token
        if len(results) > 0:
            return [x[0] for x in Counter([tok.lemma_ for tok in results if tok.pos_ in ['NOUN', 'X', 'PROPN', 'ADJ'] and not tok.is_stop and tok.is_alpha]).most_common(self.nb_tokens)]
        return []
