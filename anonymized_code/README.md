# Code for "Uniform Information Density Predicts the bǎ/SVO Alternation in Spontaneous Mandarin"

This repository contains the analysis pipeline used in the paper, from corpus extraction to the final statistical models and figures. It is an anonymized snapshot prepared for double-blind review.

## Data

The corpus used is MagicData-RAMC (Yang et al., 2022), a publicly available Mandarin conversational speech corpus. Raw corpus files are not redistributed here; they can be obtained from the original corpus provider. The scripts below expect the corpus in its original per-file transcript format unless otherwise noted.

## Pipeline

The scripts are organized by processing stage, in the order they are meant to be run.

### 1. `01_extraction/` — extracting candidate sentences

- **`extract_ba_with_context_v8.py`** — interactive script that extracts bǎ-construction candidate sentences from the corpus transcripts. Uses a three-layer strategy (HanLP dependency parsing as the primary strategy, HanLP semantic-role labelling as fallback, `jieba` as a last resort) to identify the bǎ-marker and extract NP1/NP2/VP spans, together with configurable preceding/following conversational context. Prompts for the corpus folder path; writes `<folder>_ba_with_context_v7.csv`.
- **`extract_svo.py`** — extracts SVO sentences containing the same target verbs from the corpus. Usage: `python extract_svo.py --ba_csv <ba_corpus.csv> --cts_dir <corpus_folder> --output svo_results.csv`.
- **`visualize_dep.py`** — interactive helper used during manual verification to display HanLP's dependency parse (table + ASCII tree) for a single sentence. Not part of the batch pipeline; included for transparency on how parses were manually checked.

### 2. `02_vp_classification_and_generation/` — VP classification and counterfactual generation

- **`classify_vp.py`** — assigns one or more of 13 VP-type labels (resultative, directional, quantified, aspectual markers, etc.) to each SVO candidate. Usage: `python classify_vp.py --input svo_all.csv --output svo_classified.csv`.
- **`fix_np_spans.py`** — re-extracts full NP1 (subject) and NP2 (object) spans from the dependency tree, replacing head-word-only spans. Usage: `python fix_np_spans.py --input svo_classified.csv --output svo_fixed_np.csv`.
- **`generate_ba.py`** — generates the bǎ-construction counterfactual (NP1 + 把 + NP2 + VP) for each attested SVO sentence. Usage: `python generate_ba.py --input svo_classified.csv --output svo_with_ba.csv`.
- **`generate_svo_v3.6.py`** — generates the canonical SVO counterfactual for each attested bǎ sentence, applying a construction-specific template per VP type (resultative, directional, quantified, etc.) so that the generated counterpart remains grammatical. Usage: `python generate_svo_v3.6.py --input annotated.csv --output svo_output.csv`.

### 3. `03_filtering_and_postprocessing/` — quality filtering and corrections

- **`auto_filter.py`** — automatically flags low-quality rows (sentence-medial fillers, repeated words, non-"keep" extraction decisions, bare-verb review cases). Usage: `python auto_filter.py --input svo_with_ba.csv --output svo_filtered.csv`.
- **`fix_svo_asp.py`** — regenerates the SVO string for two VP types (aspect marker 了, simple resultative) using a corrected template. Usage: `python fix_svo_asp.py --input <csv> --output <csv>`.
- **`post_process.py`** — assigns a single primary VP-type label by priority, resolves ambiguous quantifiers (些/点/个) between VP-internal and NP2-internal readings, and regenerates the bǎ counterfactual for affected rows. Usage: `python post_process.py --input svo_filtered.csv --output svo_final.csv`.

### 4. `04_surprisal_and_modeling/` — surprisal computation and statistical modelling

- **`build_model_data.py`** — merges the final annotated bǎ and SVO datasets (`ba_annotated_final.csv`, `svo_annotated_final.csv`, and, once available, their surprisal counterparts) into `model_data.csv` / `model_data_surprisal.csv`, harmonizing column names and computing derived predictors (log-lengths, givenness binary, etc.).
- **`compute_surprisal_v2.py`** — computes character-level surprisal for both members of each minimal pair using `uer/gpt2-chinese-cluecorpussmall`, with the two preceding conversational turns as left context, and derives the UID metrics (Δvariance, Δamplitude, Δmax-step, Δmean).
- **`run_model.R`** — fits the baseline mixed-effects logistic regression models (M1, M2) without the UID predictor.
- **`run_model_surprisal.R`** — fits the full models (M1, M3) including the UID predictor, plus the robustness checks across the three UID operationalizations, VIF diagnostics, and pseudo-R².

### 5. `05_figures/` — figures

- **`plot_fig1_prob_deltavariance.py`** — GAM-smoothed plot of the proportion of bǎ sentences as a function of Δvariance.

## Requirements

- Python: `pandas`, `numpy`, `torch`, `transformers`, `hanlp`, `jieba`, `matplotlib`, `pygam`
- R: `lme4`, `car`, `MuMIn`, `broom.mixed`, `ggplot2`, `dplyr`

## Use of AI assistance

We employed AI-based tools (Claude and ChatGPT) for writing and coding assistance in the development of this codebase. These tools were used in compliance with the ACL Policy on the Use of AI Writing Assistance (https://2023.aclweb.org/blog/ACL-2023-policy/). All extraction rules, generation templates, statistical models, and reported results were designed, run, and verified by the authors, who have checked the code for potential license conflicts and plagiarism.
