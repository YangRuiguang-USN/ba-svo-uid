"""
Calcul du surprisal avec context_before — version 2
Pour chaque ligne :
  - Nettoyer context_before ([+], |, etc.)
  - Calculer surprisal de la phrase cible avec context comme préfixe
  - Calculer surprisal de la phrase alternative (svo ou ba_sentence)
  - Calculer Δvariance = variance_bǎ − variance_SVO
"""

import pandas as pd
import numpy as np
import torch
import re
from transformers import BertTokenizer, GPT2LMHeadModel
import warnings
warnings.filterwarnings('ignore')

# ─── 1. Chargement du modèle ──────────────────────────────────────────────────

MODEL_NAME = "uer/gpt2-chinese-cluecorpussmall"

print("Chargement du modèle chinese-GPT2...")
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model     = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
model.eval()

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Accélération Apple Silicon (MPS) activée")
else:
    device = torch.device("cpu")
    print("CPU utilisé")

model = model.to(device)
print("Modèle prêt.\n")

# ─── 2. Fonctions utilitaires ─────────────────────────────────────────────────

def clean_context(text):
    """Nettoyer le context_before : supprimer [+], |, et autres annotations."""
    if pd.isna(text):
        return ""
    text = str(text)
    text = re.sub(r'\[\+\]', '', text)   # supprimer [+]
    text = re.sub(r'\[.*?\]', '', text)  # supprimer tout [...]
    text = re.sub(r'\|', ' ', text)      # remplacer | par espace
    text = re.sub(r'\s+', ' ', text)     # normaliser les espaces
    return text.strip()

def compute_surprisal(context, target):
    """
    Calcule le surprisal de chaque token de `target`
    en utilisant `context` comme préfixe.
    Retourne une liste de valeurs de surprisal (en bits).
    """
    context_clean = clean_context(context)
    full_text = context_clean + target if context_clean else target

    # Tokenisation
    context_tokens = tokenizer.tokenize(context_clean) if context_clean else []
    target_tokens  = tokenizer.tokenize(target)

    if not target_tokens:
        return []

    full_tokens = context_tokens + target_tokens
    full_ids    = tokenizer.convert_tokens_to_ids(full_tokens)

    # Limiter à 500 tokens de contexte (garder la fin du contexte + cible)
    # GPT-2 chinese a une limite de 512 tokens (avec [CLS])
    MAX_CONTEXT = 500
    if len(full_ids) > MAX_CONTEXT:
        # Tronquer le contexte par la gauche, garder toute la cible
        n_target = len(target_tokens)
        n_keep_context = MAX_CONTEXT - n_target
        if n_keep_context < 0:
            # Cible trop longue, tronquer aussi
            target_tokens = target_tokens[:MAX_CONTEXT]
            full_ids = tokenizer.convert_tokens_to_ids(target_tokens)
            context_tokens = []
        else:
            context_tokens = context_tokens[-n_keep_context:]
            full_ids = tokenizer.convert_tokens_to_ids(context_tokens + target_tokens)

    # Ajouter [CLS] au début
    input_ids = torch.tensor(
        [tokenizer.cls_token_id] + full_ids
    ).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_ids)
        logits  = outputs.logits  # (1, seq_len, vocab_size)

    # Calculer surprisal uniquement pour les tokens de la cible
    n_context = len(context_tokens)
    surprisals = []
    target_ids = tokenizer.convert_tokens_to_ids(target_tokens)

    for i, token_id in enumerate(target_ids):
        # Position dans la séquence complète (après [CLS] et contexte)
        pos = n_context + i
        log_probs = torch.log_softmax(logits[0, pos, :], dim=-1)
        surprisal = -log_probs[token_id].item() / np.log(2)  # en bits
        surprisals.append(surprisal)

    return surprisals


def uid_metrics(surprisal_values):
    """Métriques UID à partir d'une liste de surprisals."""
    if len(surprisal_values) < 2:
        return {
            'surprisal_mean':      np.nan,
            'surprisal_variance':  np.nan,
            'surprisal_amplitude': np.nan,
            'surprisal_max_step':  np.nan,
            'n_tokens':            len(surprisal_values),
        }
    arr = np.array(surprisal_values)
    return {
        'surprisal_mean':      float(np.mean(arr)),
        'surprisal_variance':  float(np.var(arr)),
        'surprisal_amplitude': float(np.max(arr) - np.min(arr)),
        'surprisal_max_step':  float(np.max(np.abs(np.diff(arr)))),
        'n_tokens':            len(arr),
    }


def process_pair(context, ba_sentence, svo_sentence):
    """
    Calcule les métriques pour une paire (bǎ, SVO)
    et retourne le Δvariance = variance_bǎ - variance_SVO.
    """
    surp_ba  = compute_surprisal(context, str(ba_sentence).strip())
    surp_svo = compute_surprisal(context, str(svo_sentence).strip())

    m_ba  = uid_metrics(surp_ba)
    m_svo = uid_metrics(surp_svo)

    result = {}
    for k in ['surprisal_mean', 'surprisal_variance', 'surprisal_amplitude',
              'surprisal_max_step', 'n_tokens']:
        result[f'ba_{k}']  = m_ba[k]
        result[f'svo_{k}'] = m_svo[k]

    # Δ métriques (bǎ − SVO)
    result['delta_variance']  = m_ba['surprisal_variance']  - m_svo['surprisal_variance']
    result['delta_amplitude'] = m_ba['surprisal_amplitude'] - m_svo['surprisal_amplitude']
    result['delta_max_step']  = m_ba['surprisal_max_step']  - m_svo['surprisal_max_step']
    result['delta_mean']      = m_ba['surprisal_mean']      - m_svo['surprisal_mean']

    return result

# ─── 3. Traitement BA ─────────────────────────────────────────────────────────

print("=== Traitement BA (phrase bǎ réelle → SVO générée) ===")
ba = pd.read_csv('ba_annotated_final.csv')
print(f"{len(ba)} lignes chargées")

ba_results = []
ba_errors  = []

for i, (idx, row) in enumerate(ba.iterrows()):
    if i % 100 == 0:
        print(f"  {i}/{len(ba)} traitées...")
    try:
        result = process_pair(
            context      = row['context_before'],
            ba_sentence  = row['sentence'],
            svo_sentence = row['svo'],
        )
    except Exception as e:
        ba_errors.append((idx, str(e)))
        result = {k: np.nan for k in [
            'ba_surprisal_mean', 'ba_surprisal_variance',
            'svo_surprisal_mean', 'svo_surprisal_variance',
            'delta_variance', 'delta_amplitude', 'delta_max_step', 'delta_mean'
        ]}
    ba_results.append(result)

print(f"BA terminé. {len(ba_errors)} erreurs.")
ba_out = pd.concat([ba.reset_index(drop=True),
                    pd.DataFrame(ba_results)], axis=1)
ba_out.to_csv('ba_with_surprisal.csv', index=False)
print("Exporté → ba_with_surprisal.csv\n")

# ─── 4. Traitement SVO ────────────────────────────────────────────────────────

print("=== Traitement SVO (bǎ générée → phrase SVO réelle) ===")
svo = pd.read_csv('svo_annotated_final.csv')
print(f"{len(svo)} lignes chargées")

svo_results = []
svo_errors  = []

for i, (idx, row) in enumerate(svo.iterrows()):
    if i % 100 == 0:
        print(f"  {i}/{len(svo)} traitées...")
    try:
        result = process_pair(
            context      = row['context_before'],
            ba_sentence  = row['ba_sentence'],
            svo_sentence = row['sentence'],
        )
    except Exception as e:
        svo_errors.append((idx, str(e)))
        result = {k: np.nan for k in [
            'ba_surprisal_mean', 'ba_surprisal_variance',
            'svo_surprisal_mean', 'svo_surprisal_variance',
            'delta_variance', 'delta_amplitude', 'delta_max_step', 'delta_mean'
        ]}
    svo_results.append(result)

print(f"SVO terminé. {len(svo_errors)} erreurs.")
svo_out = pd.concat([svo.reset_index(drop=True),
                     pd.DataFrame(svo_results)], axis=1)
svo_out.to_csv('svo_with_surprisal.csv', index=False)
print("Exporté → svo_with_surprisal.csv\n")

# ─── 5. Statistiques descriptives ────────────────────────────────────────────

print("=== Statistiques Δvariance ===")
print(f"BA  Δvariance — mean: {ba_out['delta_variance'].mean():.3f}, "
      f"std: {ba_out['delta_variance'].std():.3f}, "
      f"median: {ba_out['delta_variance'].median():.3f}")
print(f"SVO Δvariance — mean: {svo_out['delta_variance'].mean():.3f}, "
      f"std: {svo_out['delta_variance'].std():.3f}, "
      f"median: {svo_out['delta_variance'].median():.3f}")

print("\n=== Signe de Δvariance (bǎ plus uniforme = Δ < 0) ===")
print(f"BA  : Δ < 0 → {(ba_out['delta_variance'] < 0).sum()} / {len(ba_out)} "
      f"({(ba_out['delta_variance'] < 0).mean()*100:.1f}%)")
print(f"SVO : Δ < 0 → {(svo_out['delta_variance'] < 0).sum()} / {len(svo_out)} "
      f"({(svo_out['delta_variance'] < 0).mean()*100:.1f}%)")
