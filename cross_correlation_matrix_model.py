import json
import pickle
from pathlib import Path
from collections import defaultdict

import spacy
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm

nlp = spacy.load("en_core_web_lg")

CONFIG = json.load(open("config.json", "r"))

# adapt piclke path depending on the category
if CONFIG["CROSS_CORRELATION"]["CATEGORY"] != "":
    CONFIG[
        "PICKLE_PATH"
    ] = f"{CONFIG['PICKLE_PATH']}/{CONFIG['CROSS_CORRELATION']['CATEGORY']}"


def load_keywords_pickles() -> dict:
    """Load the keywords from the pickle files

    Returns:
        dict (dict[{model_name: {category: list}}]): dict with the keywords for each model
    """
    data = defaultdict(dict)
    split_str = "_split" if CONFIG["SPLIT"] else ""

    for pikle_file in Path(CONFIG["PICKLE_PATH"]).rglob(
        f"*_{CONFIG['NB_TOKENS']}{split_str}.pkl"
    ):
        with open(pikle_file, "rb") as f_in:
            data[pikle_file.stem][pikle_file.parent.stem] = pickle.load(f_in)

    return data


def caculate_vector_embedding(data: dict) -> dict:
    """Calculate the vector embedding for each model

    Parameters:
        data (dict): dict with the keywords for each model

    Returns:
        dict (dict[{model_name: {category: list}}]): dict with the vector embedding for each model
    """
    vect_data = {}
    for extractor_name, keywords_per_cat in tqdm(data.items()):
        vect_model = {}
        for cat, keywords in tqdm(keywords_per_cat.items(), leave=False):
            vect_model[cat] = nlp(" ".join(keywords))
        vect_data[extractor_name] = vect_model
    return vect_data


def calculate_correlation(model1: dict, model2: dict) -> float:
    """Calculate the correlation between two models by by calculating the average similarity between each pdf

    Parameters:
        model1 (dict): vector embedding of each token for model 1
        model2 (dict): vector embedding of each token for model 2

    Returns:
        float: correlation between the two models
    """
    sim_chapters = []
    for chapter_name in set(model1.keys()).intersection(set(model2.keys())):
        sim_chapters.append(model1[chapter_name].similarity(model2[chapter_name]))
    if len(sim_chapters) == 0:
        return 0
    return sum(sim_chapters) / len(sim_chapters)


def calculate_correlation_matrix(data: dict) -> list[list[float]]:
    """Calculate the correlation matrix between each model

    Parameters:
        data (dict): dict with the vector embedding for each model

    Returns:
        list[list[float]]: correlation matrix between each model"""
    matrix = []
    for _, keywords_per_cat_1 in data.items():
        row = []
        for _, keywords_per_cat_2 in data.items():
            row.append(calculate_correlation(keywords_per_cat_1, keywords_per_cat_2))
        matrix.append(row)
    return matrix


# plot the correlation matrix
def plot_correlation_matrix(corr_matrix: list[list[float]], labels):
    g = sns.clustermap(
        corr_matrix,
        cmap="viridis",
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.5},
        yticklabels=labels,
        xticklabels=labels,
        cbar_pos=(0.03, 0.03, 0.04, 0.15),
    )

    mask = np.triu(np.ones_like(corr_matrix))
    values = g.ax_heatmap.collections[0].get_array().reshape(corr_matrix.shape)
    new_values = np.ma.array(values, mask=mask)
    g.ax_heatmap.collections[0].set_array(new_values)
    g.ax_col_dendrogram.set_visible(False)
    g.ax_heatmap.set_title(
        f"Similarity of keywords extracted from cs.{CONFIG['CROSS_CORRELATION']['CATEGORY'] if CONFIG['CROSS_CORRELATION']['CATEGORY'] != '' else 'XX'}"
    )

    plt.subplots_adjust(top=1.12)
    g.ax_cbar.set_position([0.03, 0.03, 0.06, 0.15])

    cat_str = (
        f"_{CONFIG['CROSS_CORRELATION']['CATEGORY']}"
        if CONFIG["CROSS_CORRELATION"]["CATEGORY"] != ""
        else ""
    )
    plt.savefig(f"{CONFIG['CROSS_CORRELATION']['FILE_NAME_RESULT']}{cat_str}.png")


def main():
    data = load_keywords_pickles()
    vect_data = caculate_vector_embedding(data)

    corr_matrix = calculate_correlation_matrix(vect_data)
    corr_matrix = np.array(corr_matrix)

    labels = [
        CONFIG["MODEL_NAME_TO_TEXT"]["_".join(l.split("_")[:-2])]
        for l in vect_data.keys()
    ]
    plot_correlation_matrix(corr_matrix, labels)


if __name__ == "__main__":
    main()
