import os
import json
import math
import pickle
import random
import itertools
import multiprocessing
from pathlib import Path
from textwrap import wrap
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import umap
import spacy
import torch
import numpy as np
import gensim.downloader as gensim_dwnldr
from tqdm import tqdm
from matplotlib import pyplot as plt
from transformers import AutoTokenizer, BertModel, GPT2LMHeadModel
from sklearn.manifold import SpectralEmbedding, TSNE, LocallyLinearEmbedding

if torch.cuda.is_available():
    import cuml.manifold as cu_manifold

CONFIG = json.load(open("config.json", "r"))

list_model_gensim = [
    "fasttext-wiki-news-subwords-300",
    "glove-wiki-gigaword-300",
    "word2vec-google-news-300",
]
list_model_gpt = ["gpt2-large"]
list_model_bert = ["bert-large-cased", "bert-large-uncased"]
list_model_spacy = ["spacy"]

nlp = spacy.load("en_core_web_lg")
nlp.disable_pipes("tagger", "parser", "ner", "lemmatizer", "attribute_ruler", "senter")


def get_vectorizer(model_name):
    """Get the vectorizer based on the model name

    Parameters:
        model_name (str): name of the embedding model

    Returns:
        vectorizer (function): function to vectorize the data
    """
    if model_name in list_model_spacy:
        vectorizer = lambda word: nlp(word).vector
    elif model_name in list_model_gensim:
        vect_model = gensim_dwnldr.load(model_name)
        vectorizer = lambda word: vect_model.get_vector(word)
    elif model_name in list_model_gpt:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        vect_model = GPT2LMHeadModel.from_pretrained(model_name)
        vectorizer = (
            lambda word: vect_model.transformer.wte.weight[tokenizer.encode(word)]
            .detach()
            .numpy()[0]
        )
    elif model_name in list_model_bert:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        vect_model = BertModel.from_pretrained(model_name)
        vectorizer = (
            lambda word: vect_model.embeddings.word_embeddings.weight[
                tokenizer.encode(word)[1:-1]
            ]
            .detach()
            .numpy()[0]
        )

    return vectorizer


def vectorize_data(data, vectorizer):
    """Vectorize the data

    Parameters:
        data (dict): data to vectorize
        vectorizer (function): function to vectorize the data

    Returns:
        vect_data (dict): vectorized data
    """

    def vectorize_categories(
        ext_model_name: str, keywords_per_cat: dict
    ) -> tuple[str, dict]:
        """Vectorize the categories

        Parameters:
            ext_model_name (str): name of the extractor model
            keywords_per_cat (dict): keywords per category

        Returns:
            ext_model_name (str): name of the extractor model
            data (dict): vectorized data
        """
        vect_ext_data = {}
        for cat_name, keyords in tqdm(keywords_per_cat.items(), leave=False):
            vect_ext_data[cat_name] = []
            for word in keyords:
                try:
                    vect_ext_data[cat_name].append(vectorizer(word))
                except Exception:
                    pass
        return (ext_model_name, vect_ext_data)

    vect_data = {}
    with ThreadPoolExecutor(8) as pool:
        tasks = [
            pool.submit(vectorize_categories, ext_model_name, keywords_per_cat)
            for ext_model_name, keywords_per_cat in data.items()
        ]
        results = [t.result() for t in tasks]
        for ext_model_name, vect_ext_data in results:
            vect_data[ext_model_name] = vect_ext_data

    return vect_data


def subsample(data, sample_rate):
    """Subsample the data

    Parameters:
        data (dict): data to subsample
        sample_rate (float): sample rate

    Returns:
        data (dict): subsampled data
    """
    sample_size = max(int(len(data) * sample_rate), min(100, len(data)))
    return np.array(random.sample(list(data), sample_size))


def load_data():
    data = defaultdict(lambda: defaultdict(set))
    nb_pdf_cat_model = defaultdict(lambda: defaultdict(int))
    stats = defaultdict(list)

    # load hugging face pkls
    split_str = "_split" if CONFIG["SPLIT"] else ""
    for f in Path(CONFIG["PICKLE_PATH"]).rglob(
        f"*_{CONFIG['NB_TOKENS']}{split_str}.pkl"
    ):
        with open(f, "rb") as f_in:
            category = f.parent.parent.stem
            if (
                category in CONFIG["MANIFOLD"]["CATEGORIES_TO_NAME"].keys()
                and nb_pdf_cat_model[f.stem][category] < 100
            ):
                nb_pdf_cat_model[f.stem][category] += 1
                pick_load = pickle.load(f_in)
                stats[f.stem].append(len(pick_load))
                data[f.stem][category].update(pick_load)

    # print some stats
    # random words extracted by each extractor
    for k, v in data.items():
        words = set([x for v1 in v.values() for x in v1])
        print(f"{k}: {random.sample(words, min(100, len(words)))}")

    # number of keywords extracted by each extractor
    for k, v in stats.items():
        print(f"{k}: {np.mean(v):.2f} {np.std(v):.2f}", flush=True)

    return data


def store_data(data, model_name):
    """Store the data

    Parameters:
        data (dict): data to store
        model_name (str): name of the model
    """
    store = [
        data,
    ]
    file_name = f"{CONFIG['MANIFOLD']['STORE_PICKLE_PATH']}_{model_name}.pkl"
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name, "wb") as f:
        pickle.dump(store, f)


def calculate_and_plot_manifold(
    manifold,
    ext_model_name,
    vect_keywords_per_ext,
    axis,
    sub_sample_rate,
    category_to_name,
):
    """Calculate and plot the manifold for a given extractor model and manifold

    Parameters:
        manifold (tuple): manifold model
        ext_model_name (str): name of the extractor model
        vect_keywords_per_ext (dict): vectorized keywords per extractor model
        axis (matplotlib.axes._subplots.AxesSubplot): axis to plot the manifold
        sub_sample_rate (float): subsample rate
        category_to_name (dict): category to name

    Returns:
        is_skip: 0 if the manifold is plotted, 1 otherwise
    """
    vects = np.array(
        [
            vec
            for _, vects_words in vect_keywords_per_ext.items()
            for vec in subsample(vects_words, sub_sample_rate)
        ]
    )
    if len(vects) <= 5:
        return 1

    cat_names = list(vect_keywords_per_ext.keys())
    split_index = list(
        itertools.accumulate(
            [
                len(subsample(vects_words, sub_sample_rate))
                for vects_words in vect_keywords_per_ext.values()
            ],
            lambda x, y: x + y,
        )
    )
    split_range = list(zip([0] + split_index[:-1], split_index))
    axis.set_title(
        "\n".join(
            wrap(
                # remove the last 2 elements of the model name before converting to readable name
                CONFIG["MODEL_NAME_TO_TEXT"]["_".join(ext_model_name.split("_")[:-2])],
                40,
            )
        )
    )

    if len(vects) > 5000000 or "non_cuda" not in manifold:
        emb_2d = manifold["model"](**manifold["arg"]).fit_transform(vects)
    else:
        non_cuda_manifold = manifold["non_cuda"]
        emb_2d = non_cuda_manifold["model"](**non_cuda_manifold["arg"]).fit_transform(
            vects
        )

    # plot each category
    for cat, (start, end) in zip(cat_names, split_range):
        if cat not in category_to_name.keys():
            continue
        subsample_emb_2d = subsample(emb_2d[start:end], sub_sample_rate)

        if len(subsample_emb_2d) > 0:
            axis.scatter(*subsample_emb_2d.T, s=2**2, label=category_to_name[cat])
        elif len(emb_2d[start:end]) > 0:
            axis.scatter(*emb_2d[start:end].T, s=2**2, label=category_to_name[cat])
        else:
            continue

    remove_outliers_on_graph(emb_2d, axis)
    return 0


def remove_outliers_on_graph(data, axis):
    """Remove outliers on the graph axis

    Parameters:
        data (dict): data to remove outliers
        axis (pyplot.axis): axis to remove outliers
    """
    ypbot = np.nanpercentile(data.T[1, :], 5)
    yptop = np.nanpercentile(data.T[1, :], 95)
    ypad = 0.1 * (yptop - ypbot)
    ymin = ypbot - ypad
    ymax = yptop + ypad

    xpbot = np.nanpercentile(data.T[0, :], 5)
    xptop = np.nanpercentile(data.T[0, :], 95)
    xpad = 0.1 * (xptop - xpbot)
    xmin = xpbot - xpad
    xmax = xptop + xpad

    if (
        ymin != np.nan
        and ymax != np.nan
        and not math.isnan(ymin)
        and not math.isnan(ymax)
    ):
        axis.set_ylim(ymin, ymax)
    if (
        xmin != np.nan
        and xmax != np.nan
        and not math.isnan(xmin)
        and not math.isnan(xmax)
    ):
        axis.set_xlim(xmin, xmax)


def manifold_plotting(embedding_model_name, pickle_data):
    """Plot the manifold

    Parameters:
        embedding_model_name (str): name of the model
        data (dict): data to process and plot
    """

    STORE_PICKLE_PATH = CONFIG["MANIFOLD"]["STORE_PICKLE_PATH"]
    CATEGORY_TO_NAME = CONFIG["MANIFOLD"]["CATEGORIES_TO_NAME"]
    SUB_SAMPLE_RATE = CONFIG["MANIFOLD"]["SUB_SAMPLE_RATE"]

    if not os.path.exists(f"{STORE_PICKLE_PATH}_{embedding_model_name}.pkl"):
        vectorizer = get_vectorizer(embedding_model_name)
        vect_data = vectorize_data(pickle_data, vectorizer)

        store_data(vect_data, embedding_model_name)
    
    with open(f"{STORE_PICKLE_PATH}_{embedding_model_name}.pkl", "rb") as f:
        store = pickle.load(f)

    data = store[0]
    del store

    if torch.cuda.is_available():
        manifolds = [
            {
                "model": cu_manifold.TSNE,
                "name": "tsne",
                "arg": {"perplexity": 5},
                "non_cuda": {"model": TSNE, "name": "tsne", "arg": {"perplexity": 4}},
            },
            {
                "model": cu_manifold.UMAP,
                "name": "umap",
                "arg": {},
                "non_cuda": {"model": umap.UMAP, "name": "umap", "arg": {}},
            },
            {"model": SpectralEmbedding, "name": "spectral", "arg": {}},
            {
                "model": LocallyLinearEmbedding,
                "name": "linear",
                "arg": {"eigen_solver": "dense"},
            },
        ]
    else:
        manifolds = [
            {"model": TSNE, "name": "tsne", "arg": {"perplexity": 4}},
            {"model": umap.UMAP, "name": "umap", "arg": {}},
            {"model": SpectralEmbedding, "name": "spectral", "arg": {}},
            {
                "model": LocallyLinearEmbedding,
                "name": "linear",
                "arg": {"eigen_solver": "dense"},
            },
        ]

    # calculate the manifold and plot them with different models
    for manifold in tqdm(manifolds, leave=False):
        fig, ax = plt.subplots(4, 4, figsize=(11, 9))
        i = 0
        nb_skip = 0
        for ext_model_name, vect_keywords_per_ext in data.items():
            nb_skip += calculate_and_plot_manifold(
                manifold,
                ext_model_name,
                vect_keywords_per_ext,
                ax[i // 4, i % 4],
                SUB_SAMPLE_RATE,
                CATEGORY_TO_NAME,
            )
            i += 1

        # remove the plots that are not used
        for i in range(len(data.keys()) - nb_skip, len(ax.flatten())):
            ax[i // 4, i % 4].set_axis_off()

        # configure the genral plot parameters
        handles, labels = ax[0, 0].get_legend_handles_labels()
        fig.suptitle(f'2d manifold with {manifold["name"]} model', fontweight="bold")
        fig.legend(
            handles,
            labels,
            loc="outside lower center",
            title="Title: Arxiv listings",
            ncol=3,
        )
        fig.tight_layout()
        fig.subplots_adjust(top=0.92, bottom=0.15)
        os.makedirs("results/manifolds", exist_ok=True)
        plt.savefig(
            f'results/manifolds/{embedding_model_name}_{manifold["name"]}_model.png',
            dpi=200,
        )
        plt.close()


def main():
    # can speed up the process by using multiprocessing but use a lot of memory
    # with ProcessPoolExecutor(8, mp_context=multiprocessing.get_context('spawn')) as pool:
    #     tasks = [pool.submit(manifold, embedding_model_name) for embedding_model_name in (list_model_spacy + list_model_gpt + list_model_bert + list_model_gensim)]
    #     print('***** WAITING *****', flush=True)
    #     [t.result() for t in tasks]
    # print('***** DONE *****', flush=True)
    data = load_data()
    for embedding_model_name in tqdm(
        (list_model_spacy + list_model_gpt + list_model_bert + list_model_gensim)
    ):
        manifold_plotting(embedding_model_name, data)


if __name__ == "__main__":
    main()
