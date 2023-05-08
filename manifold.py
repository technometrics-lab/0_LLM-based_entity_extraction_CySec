from sklearn.manifold import SpectralEmbedding, TSNE, LocallyLinearEmbedding
import umap
import spacy
from pathlib import Path
import pickle
from matplotlib import pyplot as plt
import os
import itertools
import numpy as np
from tqdm import tqdm
from textwrap import wrap
import gensim.downloader as gensim_dwnldr
from transformers import AutoTokenizer, BertModel, GPT2LMHeadModel
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing
import torch
import math
import random

if torch.cuda.is_available():
    import cuml.manifold as cu_manifold

NB_TOKEN = 100
SPLIT = True
PIKLE_PATH = 'results/pickle_model_keyword'
STORE_PIKLE_PATH = 'results/store'
SUB_SAMPLE_RATE = 0.6

KEEP_CATGEORIES = ['NI', 'CR', 'CL', 'AI', 'DS', 'CC', 'LO', 'IT']

CATEGORIES_TO_NAME = {
    'NI': 'Network and Internet Architecture: cs.NI',
    'CR': 'Cryptography and Security: cs.CR',
    'CL': 'Computation and Language: cs.CL',
    'AI': 'Artificial Intelligence: cs.AI',
    'DS': 'Data Structures and Algorithms: cs.DS',
    'CC': 'Computational Complexity: cs.CC',
    'LO': 'Logic in Computer Science: cs.LO',
    'IT': 'Information Theory: cs.IT'
}

# Readble model name conversion
model_name_conversion = {
    'yake': 'Yake',
    'hugging_bhadresh-savani_electra-base-discriminator-finetuned-conll03-english': 'Electra',
    'hugging_asahi417_tner-xlm-roberta-base-ontonotes5': 'XLM-Roberta-base',
    'hugging_browndw_docusco-bert': 'BERT',
    'keybert': 'KeyBERT',
    'hugging_Jean-Baptiste_roberta-large-ner-english': 'Roberta',
    'hugging_yanekyuk_bert-uncased-keyword-extractor': 'BERT uncased extractor',
    'hugging_dslim_bert-large-NER': 'Bert-large',
    'hugging_xlm-roberta-large-finetuned-conll03-english': 'XLM-Roberta-large',
    'hugging_elastic_distilbert-base-uncased-finetuned-conll03-english': 'DistilBert uncased',
    'hugging_yanekyuk_bert-uncased-keyword-discriminator': 'BERT uncased discriminator',
    'hugging_Jorgeutd_bert-large-uncased-finetuned-ner': 'BERT large uncased',
    'hugging_ml6team_keyphrase-extraction-kbir-kpcrowd': 'kbir-kpcrowd',
    'en_core_web_trf': 'Spacy Transformer',
    'en_core_web_lg': 'Spacy Large',
    'hugging_ml6team_keyphrase-extraction-kbir-inspec': 'kbir-inspec',
}

list_model_gensim = ['fasttext-wiki-news-subwords-300', 'glove-wiki-gigaword-300', 'word2vec-google-news-300']
list_model_gpt = ['gpt2-large']
list_model_bert = ['bert-large-cased', 'bert-large-uncased']
list_model_spacy = ['spacy']

nlp = spacy.load('en_core_web_lg')
nlp.disable_pipes('tagger', 'parser', 'ner', 'lemmatizer', 'attribute_ruler', 'senter')

# select the vectorizer based on the model name
# model_name (str): name of the model
def get_vectorizer(model_name):
    if model_name in list_model_spacy:
        vectorizer = lambda x: nlp(x).vector
    elif model_name in list_model_gensim:
        vect_model = gensim_dwnldr.load(model_name)
        vectorizer = lambda x: vect_model.get_vector(x)
    elif model_name in list_model_gpt:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        vect_model = GPT2LMHeadModel.from_pretrained(model_name)
        vectorizer = lambda x: vect_model.transformer.wte.weight[tokenizer.encode(x)].detach().numpy()[0]
    elif model_name in list_model_bert:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        vect_model = BertModel.from_pretrained(model_name)
        vectorizer = lambda x: vect_model.embeddings.word_embeddings.weight[tokenizer.encode(x)[1:-1]].detach().numpy()[0]

    return vectorizer

# vectorize the data
# data (dict): data to vectorize
# vectorizer (function): function to vectorize the data
def vectorize_data(data, vectorizer):
    def vectorize_categories(k1, v1):
        data_inside = {}
        for k2, v2 in tqdm(v1.items(), leave=False):
            data_inside[k2] = []
            for x in v2:
                try:
                    data_inside[k2].append(vectorizer(x))
                except Exception as e:
                    pass
        return (k1, data_inside)

    data_temp = {}
    with ThreadPoolExecutor(8) as pool:
        tasks = [pool.submit(vectorize_categories, k1, v1) for k1, v1 in data.items()]
        results = [t.result() for t in tasks]
        for (k, v) in results:
            data_temp[k] = v

    return data_temp

# plot the manifold
# model_name (str): name of the model
def manifold(model_name):
    if not os.path.exists(f'{STORE_PIKLE_PATH}_{model_name}.pkl'):
        data = defaultdict(lambda: defaultdict(list))

        #load hugging face pkls
        split_str = '_split' if SPLIT else ''
        for f in Path(PIKLE_PATH).rglob(f'*_{NB_TOKEN}{split_str}.pkl'):
            with open(f, 'rb') as f_in:
                category = f.parent.parent.stem
                if category in KEEP_CATGEORIES:
                    data[f.stem][category].extend(pickle.load(f_in))

        vectorizer = get_vectorizer(model_name)
        data = vectorize_data(data, vectorizer)        
        
        store = [data, ]
        file_name = f'{STORE_PIKLE_PATH}_{model_name}.pkl'
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        with open(file_name, 'wb') as f:
            pickle.dump(store, f)
    else:
        with open(f'{STORE_PIKLE_PATH}_{model_name}.pkl', 'rb') as f:
            store = pickle.load(f)

    data = store[0]
    del store

    if torch.cuda.is_available():
        manifolds = [{'model':cu_manifold.TSNE, 'name': 'tsne', 'arg':{'perplexity': 5}}, {'model':cu_manifold.UMAP, 'name': 'umap', 'arg':{}}, {'model':SpectralEmbedding, 'name': 'spectral', 'arg':{}}, {'model':LocallyLinearEmbedding, 'name': 'linear', 'arg':{'eigen_solver': 'dense'}}]
    else:
        manifolds = [{'model':SpectralEmbedding, 'name': 'spectral', 'arg':{}}, {'model':umap.UMAP, 'name': 'umap', 'arg':{}}, {'model':LocallyLinearEmbedding, 'name': 'linear', 'arg':{'eigen_solver': 'dense'}}, {'model':TSNE, 'name': 'tsne', 'arg':{'perplexity': 4}}]

    # calculate the manifold and plot them with different models
    nb_skip = 0
    for manifold in tqdm([manifolds[0]], leave=False):
        fig, ax = plt.subplots(4, 4, figsize=(25, 20))
        i = 0
        for k, v in data.items():
            vects = np.array([x for _, v1 in v.items() for x in v1])

            chapter_name = list(v.keys())
            split_index = list(itertools.accumulate([len(v1) for v1 in v.values()], lambda x, y: x+y))
            split_range = list(zip([0] + split_index[:-1], split_index))
            ax[i//4, i%4].set_title('\n'.join(wrap(model_name_conversion["_".join(k.split("_")[:-2])], 40)))
            if len(vects) < 4:
                nb_skip += 1
                continue

            emb_2d = manifold['model'](**manifold['arg']).fit_transform(vects)

            for cn, (s, e) in zip(chapter_name, split_range):
                if cn not in KEEP_CATGEORIES:
                    continue
                subsabmple = np.array(random.sample(list(emb_2d[s:e]), int(len(emb_2d[s:e])*SUB_SAMPLE_RATE)))
                if len(subsabmple) > 0:
                    ax[i//4, i%4].scatter(*subsabmple.T, s=2**2, label=CATEGORIES_TO_NAME[cn])
                elif len(emb_2d[s:e]) > 0:
                    ax[i//4, i%4].scatter(*emb_2d.T, s=2**2, label=CATEGORIES_TO_NAME[cn])

            # remove outliers
            ypbot = np.nanpercentile(emb_2d.T[1, :], 1)
            yptop = np.nanpercentile(emb_2d.T[1, :], 99)
            ypad = 0.2*(yptop - ypbot)
            ymin = ypbot - ypad
            ymax = yptop + ypad

            xpbot = np.nanpercentile(emb_2d.T[0, :], 1)
            xptop = np.nanpercentile(emb_2d.T[0, :], 99)
            xpad = 0.2*(xptop - xpbot)
            xmin = xpbot - xpad
            xmax = xptop + xpad

            if ymin != np.nan and ymax != np.nan and not math.isnan(ymin) and not math.isnan(ymax):
                ax[i//4, i%4].set_ylim(ymin, ymax)
            if xmin != np.nan and xmax != np.nan and not math.isnan(xmin) and not math.isnan(xmax):
                ax[i//4, i%4].set_xlim(xmin, xmax)
            i += 1

        for i in range(len(data.keys())-nb_skip, len(ax.flatten())):
            ax[i//4, i%4].set_axis_off()

        handles, labels = ax[0,0].get_legend_handles_labels()
        fig.suptitle(f'2d embedding with {manifold["name"]} model', fontweight='bold')
        fig.legend(handles, labels, loc='outside center right', title='Title: Arxiv listings')
        fig.tight_layout()
        fig.subplots_adjust(top=0.92, right=0.85)
        os.makedirs('results/manifolds', exist_ok=True)
        plt.savefig(f'results/manifolds/{model_name}_{manifold["name"]}_model.png', dpi=300)
        plt.close()

def main():
    #can speed up the process by using multiprocessing but use a lot of memory
    # with ProcessPoolExecutor(8, mp_context=multiprocessing.get_context('spawn')) as pool:
    #     tasks = [pool.submit(manifold, model_name) for model_name in (list_model_spacy + list_model_gpt + list_model_bert + list_model_gensim)]
    #     print('***** WAITING *****', flush=True)
    #     [t.result() for t in tasks]
    # print('***** DONE *****', flush=True)
    for model_name in tqdm((list_model_spacy + list_model_gpt + list_model_bert + list_model_gensim)):
        manifold(model_name)

if __name__ == '__main__':
    main()
