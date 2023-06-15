import os
import json
import math
import time
import pickle
import random
import itertools
import multiprocessing
from pathlib import Path
from textwrap import wrap
from collections import OrderedDict, defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import umap
import spacy
import torch
import numpy as np
import seaborn as sns
import gensim.downloader as gensim_dwnldr
from tqdm import tqdm
from matplotlib import pyplot as plt
from transformers import AutoTokenizer, BertModel, GPT2LMHeadModel
from sklearn.manifold import SpectralEmbedding, TSNE, LocallyLinearEmbedding

if torch.cuda.is_available():
    import cuml.manifold as cu_manifold

from flowcontrol.registration import SavePaths

CONFIG = json.load(open("src/flowcontrol/config.json", "r"))

# fix the random seed for reproducibility. Can be not exact since some algortihms work in parallel
RANDOM_SEED = random.randint(0, 4294967295)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

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
        for cat_name, keyords in tqdm(
            keywords_per_cat.items(), leave=False, ascii=False
        ):
            vect_ext_data[cat_name] = []
            for word in keyords:
                try:
                    vect_ext_data[cat_name].append(vectorizer(word))
                except Exception:
                    pass
        return (ext_model_name, vect_ext_data)

    vect_data = OrderedDict()
    for name in CONFIG["MODEL_NAME_TO_TEXT"].values():
        vect_data[name] = None
    with ThreadPoolExecutor(8) as pool:
        tasks = [
            pool.submit(vectorize_categories, ext_model_name, keywords_per_cat)
            for ext_model_name, keywords_per_cat in data.items()
        ]
        results = [t.result() for t in tasks]
        for ext_model_name, vect_ext_data in results:
            vect_data[ext_model_name] = vect_ext_data
    vect_data = {key: vects for key, vects in vect_data.items() if vects is not None}
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


def load_data(active_savepath):
    """Load the data

    Parameters:
        active_savepath (SavePaths): active savepath

    Returns:
        data (dict): data
    """
    data = defaultdict(lambda: defaultdict(set))
    nb_pdf_cat_model = defaultdict(lambda: defaultdict(int))
    stats = defaultdict(list)

    # load pkls
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
                active_savepath.logging(
                    f"Loaded {f.stem} {category}: {len(pick_load)} {pick_load[:10]}"
                )

    # print some stats
    # random words extracted by each extractor
    for k, v in data.items():
        words = set([x for v1 in v.values() for x in v1])
        print(
            f'{CONFIG["MODEL_NAME_TO_TEXT"]["_".join(k.split("_")[:-2])]}: {random.sample(words, min(100, len(words)))}'
        )

    # number of keywords extracted by each extractor
    for k, v in stats.items():
        print(
            f'{CONFIG["MODEL_NAME_TO_TEXT"]["_".join(k.split("_")[:-2])]}: {np.mean(v):.2f} {np.std(v):.2f}',
            flush=True,
        )

    return data


def store_data(data, file_path):
    """Store the data

    Parameters:
        data (dict): data to store
        file_path (str): path to store the data
    """
    store = [
        data,
    ]
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "wb") as f:
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
                20,
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


def distance(x, y):
    """Calculate the distance between 2 vectors

    Parameters:
        x (np.array): vector x
        y (np.array): vector y

    Returns:
        float: distance between x and y
    """
    return np.linalg.norm(x - y)


def get_centroid(data):
    """Calculate the centroid of the vectors

    Parameters:
        data (np.array): list of vectors

    Returns:
        np.array: centroid of the vectors
    """
    return np.median(data, axis=0), np.std(data, axis=0)


def mean_distance(data, centroid):
    """Calculate the mean distance between the vectors and the centroid

    Parameters:
        data (np.array): list of vectors
        centroid (np.array): centroid of the vectors

    Returns:
        np.array: mean distance between the vectors and the centroid
    """
    distances = [distance(x, centroid) for x in data]
    return np.median(distances, axis=0), np.std(distances, axis=0)


def pairwise_distance(cat_cent, cat_std, rem_cent, rem_std):
    """Calculate the pairwise distance between to sets of vectors

    Parameters:
        cat_data (np.array): list of vectors in category
        rem_data (np.array): list of vectors int the other categories

    Returns:
        np.array: pairwise distance between the two sets of vectors
    """
    return np.sqrt(cat_std**2 + rem_std**2) / distance(cat_cent, rem_cent)


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


def plot_cluster_coefs(active_savepath, embedding_model_name, data, category_to_name):
    """Plot the cluster coefficients of the embeddings

    Parameters:
        active_savepath (str): path to save the plot
        embedding_model_name (str): name of the embedding model
        data (dict): data to plot
        category_to_name (dict): mapping from category to name
    """
    for ext_model_name, vect_keywords_per_ext in data.items():
        start_time = time.time()
        print("Start calculating clust_coeff", flush=True)
        cats_data = dict()
        for cat, cat_data in vect_keywords_per_ext.items():
            cat_data = subsample(cat_data, 0.5)
            centroid, cent_std = get_centroid(cat_data)
            cat_dist, cat_std = mean_distance(cat_data, centroid)
            cats_data[cat] = (centroid, cent_std, cat_dist, cat_std)

        clust_coeff = []
        for cat1, data1 in cats_data.items():
            line = []
            for cat2, data2 in cats_data.items():
                line.append(
                    data1[2]
                    / pairwise_distance(
                        data1[0], data1[3], data2[0], data2[3]
                    )
                )
            clust_coeff.append(line)
        print(f"End calculating clust_coeff: {time.time() - start_time}", flush=True)

        clust_coeff = np.array(clust_coeff)

        labels_short = []
        labels_long = []
        for cat in vect_keywords_per_ext.keys():
            labels_short.append(f"cs.{cat}")
            labels_long.append(f"{category_to_name[cat]}")

        g = sns.clustermap(
            clust_coeff,
            cmap="viridis",
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.5},
            yticklabels=labels_long,
            xticklabels=labels_short,
            cbar_pos=(0.03, 0.03, 0.04, 0.15),
            vmin=0.65,
            vmax=0.75,
        )

        mask = np.triu(np.ones_like(clust_coeff))
        values = g.ax_heatmap.collections[0].get_array().reshape(clust_coeff.shape)
        new_values = np.ma.array(values, mask=mask)
        g.ax_heatmap.collections[0].set_array(new_values)
        g.ax_col_dendrogram.set_visible(False)
        g.ax_heatmap.set_title(
            f"Cluster coefficient using {embedding_model_name}, extract\n with {CONFIG['MODEL_NAME_TO_TEXT']['_'.join(ext_model_name.split('_')[:-2])]}"
        )

        plt.subplots_adjust(top=1.12, bottom=0.15)
        g.ax_cbar.set_position([0.03, 0.03, 0.04, 0.15])

        os.makedirs(f"{active_savepath.result}/cluster_coeff", exist_ok=True)
        plt.savefig(
            f"{active_savepath.result}/cluster_coeff/cluster_coeff_{embedding_model_name}_{ext_model_name}.png"
        )
        plt.close()


def manifold_plotting(active_savepath, embedding_model_name, pickle_data):
    """Plot the manifold

    Parameters:
        active_savepath (object): active savepath
        embedding_model_name (str): name of the model
        pickle_data (dict): data to process and plot
    """

    CATEGORY_TO_NAME = CONFIG["MANIFOLD"]["CATEGORIES_TO_NAME"]
    SUB_SAMPLE_RATE = CONFIG["MANIFOLD"]["SUB_SAMPLE_RATE"]

    file_path = (
        f"{active_savepath.result}/pickle_manifold/store_{embedding_model_name}.pkl"
    )

    if not os.path.exists(file_path):
        vectorizer = get_vectorizer(embedding_model_name)
        vect_data = vectorize_data(pickle_data, vectorizer)

        store_data(vect_data, file_path)

    with open(file_path, "rb") as f:
        store = pickle.load(f)

    data = store[0]
    del store

    plot_cluster_coefs(active_savepath, embedding_model_name, data, CATEGORY_TO_NAME)

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
    for manifold in tqdm(manifolds, leave=False, ascii=False):
        manifold["arg"].update({"random_state": RANDOM_SEED})
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
        path = f"{active_savepath.result}/manifolds/"
        os.makedirs(path, exist_ok=True)
        fig.savefig(
            f'{path}{embedding_model_name}_{manifold["name"]}_model.png',
            dpi=200,
        )
        plt.close(fig)


def main():
    # can speed up the process by using multiprocessing but use a lot of memory not totally up to date
    # with ProcessPoolExecutor(8, mp_context=multiprocessing.get_context('spawn')) as pool:
    #     tasks = [pool.submit(manifold, embedding_model_name) for embedding_model_name in (list_model_spacy + list_model_gpt + list_model_bert + list_model_gensim)]
    #     print('***** WAITING *****', flush=True)
    #     [t.result() for t in tasks]
    # print('***** DONE *****', flush=True)
    active_savepath = SavePaths("manifold")
    print(f"RANDOM SEED: {RANDOM_SEED}")
    data = load_data(active_savepath)
    for embedding_model_name in tqdm(
        (list_model_spacy + list_model_gpt + list_model_bert + list_model_gensim),
        ascii=False,
    ):
        active_savepath.logging(f"Manifold for {embedding_model_name}")
        manifold_plotting(active_savepath, embedding_model_name, data)
    active_savepath.move_result_to_final_path()


if __name__ == "__main__":
    main()
