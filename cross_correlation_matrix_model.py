from pathlib import Path
import pickle
import spacy
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import defaultdict

nlp = spacy.load('en_core_web_lg')

# number of token to extract
NB_TOKEN = 100
# if the text was split or not
SPLIT = True
PIKLE_PATH = 'results/pickle_model_keyword'
FILE_NAME_RESULT = 'results/model_correlation_'
# if a category is specified, only the data for this category will be loaded
# example: CATEGORY = 'CR'
CATEGORY = None

#adapt piclke path depending on the category
if CATEGORY is not None:
    PIKLE_PATH = f'{PIKLE_PATH}/{CATEGORY}'

# Readable model name conversion
model_name = {
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

data = defaultdict(dict)

#load hugging face pkls
split_str = '_split' if SPLIT else ''
for f in Path(PIKLE_PATH).rglob(f'*_{NB_TOKEN}{split_str}.pkl'):
    with open(f, 'rb') as f_in:
        data[f.stem][f.parent.stem] = pickle.load(f_in)

#calculate the vector embedding for each model
vect_data = {}
for k,v in tqdm(data.items()):
    vect_model = {}
    for k2,v2 in tqdm(v.items(), leave=False):
        vect_model[k2] = nlp(' '.join(v2))
    vect_data[k] = vect_model

# calculate the correlation between two models
# calculate the average similarity between each pdf
# model1: vector embedding of each tocken for model 1
# model2: vector embedding of each tocken for model 2
def calculate_correlation(model1, model2):
    sim_chapters = []
    for k in set(model1.keys()).intersection(set(model2.keys())):
        sim_chapters.append(model1[k].similarity(model2[k]))
    return sum(sim_chapters)/len(sim_chapters)

# calculate the correlation between each model
# data: vector embedding of each tocken for each model
def calculate_correlation_matrix(data):
    matrix = []
    for _,v1 in data.items():
        row = []
        for _,v2 in data.items():
            row.append(calculate_correlation(v1, v2))
        matrix.append(row)
    return matrix

# calculate the correlation matrix
corr_matrix = calculate_correlation_matrix(vect_data)
corr_matrix = np.array(corr_matrix)

# plot the correlation matrix
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
labels = [model_name["_".join(l.split("_")[:-2])] for l in vect_data.keys()]
g = sns.clustermap(corr_matrix, cmap='viridis', square=True, linewidths=.5, cbar_kws={"shrink": .5}, yticklabels=labels, xticklabels=labels, cbar_pos=(0.03, 0.03, 0.04, 0.15))

mask = np.triu(np.ones_like(corr_matrix))
values = g.ax_heatmap.collections[0].get_array().reshape(corr_matrix.shape)
new_values = np.ma.array(values, mask=mask)
g.ax_heatmap.collections[0].set_array(new_values)
g.ax_col_dendrogram.set_visible(False)
g.ax_heatmap.set_title(f"Similarity of keywords extracted from cs.{CATEGORY if CATEGORY is not None else 'XX'}")

plt.subplots_adjust(top=1.12)
g.ax_cbar.set_position([0.03, 0.03, 0.06, 0.15])

cat_str = f'_{CATEGORY}' if CATEGORY is not None else ''
plt.savefig(f'{FILE_NAME_RESULT}{cat_str}.png')
