

# !pip install evaluate==0.4.3 bert_score==0.3.13
import os 
from pathlib import Path

import numpy as np

import pandas as pd
import re

from evaluate import load
from tqdm.auto import tqdm

from typing import List

data_path = Path('../../data')

input_file = data_path / 'intermediate' / 'cmp_translations_sample_translated_translation_pairs_samples.tsv'
output_file = data_path / 'results' / 'translation_similarity' / 'cmp_translations_sample_bertscores.tsv'

df = pd.read_csv(input_file, sep='\t')

text_cols = ['text', 'translation_a', 'translation_b']
df.loc[:, text_cols] = df.loc[:, text_cols].map(lambda t: re.sub(r'"+', '"', t) if isinstance(t, str) else None)

bertscore = load("bertscore")
bertscore.seed = 42

def compute_bertscore_agreement(
        df: pd.DataFrame, 
        ref: str,
        comparisons: List[str],
        key_cols: List[str], 
        **kwargs
    ):
    assert ref in df.columns
    if isinstance(comparisons, str):
        comparisons = [comparisons]
    assert all(c in df.columns for c in comparisons)
    
    out = []
    
    for c in tqdm(comparisons):
        idxs = np.logical_and(df[ref].notnull(), df[c].notnull())
        
        scores = bertscore.compute(
            predictions=df.loc[idxs, ref].to_list(), 
            references=df.loc[idxs, c].to_list(), 
            lang='en',
            **kwargs
        )
        out.append(df[key_cols].copy())
        out[-1]['text_a'] = df[ref].to_list()
        out[-1]['text_b'] = df[c].to_list()
        out[-1].loc[idxs, 'bertscore_f1'] = scores['f1']
        out[-1].loc[idxs, 'bertscore_precision'] = scores['precision']
        out[-1].loc[idxs, 'bertscore_recall'] = scores['recall']
    return pd.concat(out)


import torch
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

cols = ['other_model', 'qs_id', 'lang', 'text']
scores = compute_bertscore_agreement(
    df=df,
    ref='translation_a',
    comparisons=['translation_b'],
    key_cols=cols,
    device=device,
    batch_size=64,
)

out = scores.merge(df[cols], how='right')

os.makedirs(output_file.parent, exist_ok=True)
out.to_csv(output_file, sep='\t', float_format='%0.6f', encoding='utf-8', index=False)


