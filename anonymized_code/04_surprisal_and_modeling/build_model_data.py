"""
Construction de model_data.csv et model_data_surprisal.csv
À placer dans le même dossier que :
  - ba_annotated_final.csv
  - svo_annotated_final.csv
  - ba_with_surprisal.csv
  - svo_with_surprisal.csv
"""

import pandas as pd
import numpy as np

# ─── 1. Chargement ────────────────────────────────────────────────────────────

ba  = pd.read_csv('ba_annotated_final.csv')
svo = pd.read_csv('svo_annotated_final.csv')

print(f"BA  chargé : {len(ba)} lignes")
print(f"SVO chargé : {len(svo)} lignes")
print(f"BA  verb_class : {ba['verb_class'].value_counts().to_dict()}")
print(f"SVO verb_class : {svo['verb_class'].value_counts().to_dict()}")

# ─── 2. Harmonisation ─────────────────────────────────────────────────────────

ba['verbal_modifier_type']  = ba['verbal_modifier_type'].str.replace(' ', '_')
svo['verbal_modifier_type'] = svo['verbal_modifier_type'].str.replace(' ', '_')
svo = svo.rename(columns={'matched_verb': 'V_core'})

# ─── 3. Transformations ───────────────────────────────────────────────────────

def prepare(df, construction):
    df = df.copy()
    df['y']                 = 1 if construction == 'ba' else 0
    df['NP2_length_log']    = np.log(df['NP2_length_raw'])
    df['VP_length_log']     = np.log(df['VP_length'])
    df['NP2_givenness_bin'] = (df['NP2_givenness'] == 'given').astype(int)
    df['file_id']           = df['source_file'].str.replace('.txt', '', regex=False)
    df['construction']      = construction
    return df

ba_prep  = prepare(ba,  'ba')
svo_prep = prepare(svo, 'svo')

# ─── 4. Colonnes du modèle ────────────────────────────────────────────────────

COLS = [
    'source_file', 'file_id', 'speaker_id',
    'verbal_modifier_type', 'NP1_pronominality',
    'NP2_givenness', 'NP2_givenness_bin',
    'NP2_length_raw', 'NP2_length_log',
    'VP_length', 'VP_length_log',
    'verb_class', 'V_core',
    'construction', 'y', 'sentence',
]

# ─── 5. model_data.csv (sans surprisal) ───────────────────────────────────────

merged = pd.concat([ba_prep[COLS], svo_prep[COLS]], ignore_index=True)

model_vars = ['verbal_modifier_type', 'NP2_length_log', 'NP1_pronominality',
              'VP_length_log', 'verb_class', 'NP2_givenness_bin',
              'y', 'V_core', 'speaker_id']
complete = merged.dropna(subset=model_vars)

print(f"\n── model_data.csv ──")
print(f"Lignes : {len(complete)} (ba: {complete['y'].sum()}, svo: {(complete['y']==0).sum()})")
print("NaN par variable :")
for v in model_vars:
    n = complete[v].isna().sum()
    print(f"  {v}: {'OK' if n==0 else n}")

complete.to_csv('model_data.csv', index=False)
print("Exporté → model_data.csv")

# ─── 6. model_data_surprisal.csv (avec surprisal) ────────────────────────────

SURP_COLS = ['delta_variance', 'delta_amplitude', 'delta_max_step', 'delta_mean',
             'ba_surprisal_variance', 'svo_surprisal_variance']

ba_surp  = pd.read_csv('ba_with_surprisal.csv')
svo_surp = pd.read_csv('svo_with_surprisal.csv')

# Dédupliquer sur la clé de merge
ba_surp_slim  = ba_surp[['source_file','speaker_id','sentence'] + SURP_COLS].drop_duplicates(
    subset=['source_file','speaker_id','sentence'], keep='first')
svo_surp_slim = svo_surp[['source_file','speaker_id','sentence'] + SURP_COLS].drop_duplicates(
    subset=['source_file','speaker_id','sentence'], keep='first')

# Merge
ba_s  = complete[complete['y']==1].merge(
    ba_surp_slim, on=['source_file','speaker_id','sentence'], how='left')
svo_s = complete[complete['y']==0].merge(
    svo_surp_slim, on=['source_file','speaker_id','sentence'], how='left')

merged_s = pd.concat([ba_s, svo_s], ignore_index=True)
merged_s = merged_s.dropna(subset=['delta_variance'])

print(f"\n── model_data_surprisal.csv ──")
print(f"Lignes : {len(merged_s)} (ba: {merged_s['y'].sum()}, svo: {(merged_s['y']==0).sum()})")
print(f"NaN delta_variance : {merged_s['delta_variance'].isna().sum()}")
print(f"verb_class : {merged_s['verb_class'].value_counts().to_dict()}")

merged_s.to_csv('model_data_surprisal.csv', index=False)
print("Exporté → model_data_surprisal.csv")
