from pathlib import Path
import re
import pandas as pd

data_path = Path('../../data')
fp = data_path / 'datasets' / 'classifier_finetuning' / 'cmp_translations_sample_translated.tsv'
df = pd.read_csv(fp, sep='\t')

text_cols = ['text', 'text_mt_deepl', 'text_mt_m2m_100_1.2b', 'text_mt_opus-mt']
df.loc[:, text_cols] = df.loc[:, text_cols].map(lambda t: re.sub(r'"+', "", t) if isinstance(t, str) else None)

df = df.groupby('lang').sample(n=500, random_state=42)

df = df[['lang', 'qs_id']+text_cols]

df = df.melt(id_vars=['lang', 'qs_id', 'text', 'text_mt_deepl'], var_name='other_model', value_name='translation_b')

df.rename(columns={'text_mt_deepl': 'translation_a'}, inplace=True)

fp = data_path / 'intermediate' / 'cmp_translations_sample_translated_translation_pairs_samples.tsv'
df.to_csv(fp, sep='\t', index=False)
