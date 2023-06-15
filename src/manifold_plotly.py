import io
import os
import json
import math
import pickle
import random
import itertools
import multiprocessing
from pathlib import Path
from collections import OrderedDict, defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import umap
import spacy
import torch
import numpy as np
import pandas as pd
import plotly.express as px
import gensim.downloader as gensim_dwnldr
from tqdm import tqdm
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
                    vect_ext_data[cat_name].append((vectorizer(word), word))
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
    return random.sample(list(data), sample_size)


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
        print(f"{k}: {random.sample(words, min(100, len(words)))}")

    # number of keywords extracted by each extractor
    for k, v in stats.items():
        print(f"{k}: {np.mean(v):.2f} {np.std(v):.2f}", flush=True)

    return data


def calculate_and_plot_manifold(
    manifold,
    vect_keywords_per_ext,
    sub_sample_rate,
    category_to_name,
):
    """Calculate and plot the manifold for a given extractor model and manifold

    Parameters:
        manifold (tuple): manifold model
        vect_keywords_per_ext (dict): vectorized keywords per extractor model
        sub_sample_rate (float): subsample rate
        category_to_name (dict): category to name

    Returns:
        is_skip: 0 if the manifold is plotted, 1 otherwise
    """
    vects = [
        vec
        for _, vects_words in vect_keywords_per_ext.items()
        for vec in subsample(vects_words, sub_sample_rate)
    ]
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

    if len(vects) > 5000000 or "non_cuda" not in manifold:
        emb_2d = manifold["model"](**manifold["arg"]).fit_transform(
            np.array([vect for vect, _ in vects])
        )
    else:
        non_cuda_manifold = manifold["non_cuda"]
        emb_2d = non_cuda_manifold["model"](**non_cuda_manifold["arg"]).fit_transform(
            np.array([vect for vect, _ in vects])
        )
    emb_2d = list(zip(emb_2d, [word for _, word in vects]))

    # plot each category
    ten_biggest_cat = sorted(
        [(cat, len(vects_words)) for cat, vects_words in vect_keywords_per_ext.items()],
        key=lambda x: x[1],
        reverse=True,
    )[:10]
    ten_biggest_cat = [cat for cat, _ in ten_biggest_cat]

    cat2index = dict()
    list_printed_words = []
    fig_data = pd.DataFrame(columns=["x", "y", "word", "category"])

    for cat, (start, end) in zip(cat_names, split_range):
        if cat not in ten_biggest_cat:
            continue
        subsample_emb_2d = subsample(emb_2d[start:end], sub_sample_rate)
        words = [word for _, word in subsample_emb_2d]
        cat2index[category_to_name[cat]] = len(list_printed_words)
        list_printed_words.extend(words)
        subsample_emb_2d = np.array([emb for emb, _ in subsample_emb_2d])

        if len(subsample_emb_2d) > 0:
            df = pd.DataFrame()
            x, y = subsample_emb_2d.T
            df["x"] = x
            df["y"] = y
            df["word"] = words
            df["category"] = category_to_name[cat]
        elif len(emb_2d[start:end]) > 0:
            scatt_data = np.array([vect for vect, _ in emb_2d[start:end]])
            words = [word for _, word in emb_2d[start:end]]
            df = pd.DataFrame()
            x, y = scatt_data.T
            df["x"] = x
            df["y"] = y
            df["word"] = words
            df["category"] = category_to_name[cat]
        else:
            continue
        fig_data = pd.concat([fig_data, df], ignore_index=True)

    fig = px.scatter(
        fig_data,
        x="x",
        y="y",
        color="category",
        hover_data="word",
        opacity=0.7,
        labels={"category": "Arxiv listings"},
        custom_data="word",
    )
    fig = remove_outliers_on_graph(np.array([vect for vect, _ in emb_2d]), fig)

    return fig


def remove_outliers_on_graph(data, fig):
    """Remove outliers on the graph axis

    Parameters:
        data (dict): data to remove outliers
        fig (plotly.graph_objects.Figure): figure to update
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
        fig.update_yaxes(range=[ymin, ymax])
    if (
        xmin != np.nan
        and xmax != np.nan
        and not math.isnan(xmin)
        and not math.isnan(xmax)
    ):
        fig.update_xaxes(range=[xmin, xmax])
    return fig


def manifold_plotting(active_savepath, embedding_model_name, pickle_data):
    """Plot the manifold

    Parameters:
        active_savepath (object): active savepath
        embedding_model_name (str): name of the model
        pickle_data (dict): data to process and plot
    """

    CATEGORY_TO_NAME = CONFIG["MANIFOLD"]["CATEGORIES_TO_NAME"]
    SUB_SAMPLE_RATE = CONFIG["MANIFOLD"]["SUB_SAMPLE_RATE"]

    vectorizer = get_vectorizer(embedding_model_name)
    vect_data = vectorize_data(pickle_data, vectorizer)
    data = vect_data

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
    for manifold in tqdm(manifolds[:2], leave=False, ascii=False):
        manifold["arg"].update({"random_state": RANDOM_SEED})
        for ext_model_name, vect_keywords_per_ext in data.items():
            fig = calculate_and_plot_manifold(
                manifold,
                vect_keywords_per_ext,
                SUB_SAMPLE_RATE,
                CATEGORY_TO_NAME,
            )
            fig.update_layout(
                title=f'2d manifold with {manifold["name"]} model embedded with {embedding_model_name} using {CONFIG["MODEL_NAME_TO_TEXT"]["_".join(ext_model_name.split("_")[:-2])]}'
            )
            fig.update_layout(plot_bgcolor="white")
            fig.update_xaxes(
                mirror=True,
                ticks="outside",
                showline=True,
                linecolor="black",
                gridcolor="lightgrey",
            )
            fig.update_yaxes(
                mirror=True,
                ticks="outside",
                showline=True,
                linecolor="black",
                gridcolor="lightgrey",
            )
            fig.update_traces(hovertemplate="noun: %{customdata}<extra></extra>")

            path = f"{active_savepath.result}/freq_word_arxiv/"
            os.makedirs(path, exist_ok=True)
            fig.write_html(
                f'{path}{embedding_model_name}_{manifold["name"]}_{ext_model_name}_model.html'
            )


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
