# ACROS: Sense Representations are Inducible Interfaces

Hugging Face collection (models/checkpoints): https://huggingface.co/collections/jcblaise/acros

Paper: https://arxiv.org/abs/2605.28669

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

## Data setup

```bash
bash scripts/download_data.sh
```

Most scripts also download required datasets lazily if not already cached.

## Repository layout

- `run_sense_init.py`, `run_sense_induction.py`: ACROS sense initialization and induction training
- `run_adaptation.py`: SENSiA adaptation
- `run_post_adaptation_eval.py`: intrinsic/downstream post-adaptation evaluation
- `run_flores_ppl.py`: FLORES sentence-level PPL evaluation
- `evals/run_wsd.py`: unified WSD entrypoint
- `evals/run_steering.py`: unified CoInCo steering entrypoint
- `evals/run_generation.py`: unified XL-Sum generation/report entrypoint
- `evals/run_significance.py`: significance/CI entrypoint

## Main experiment commands

### 1) WSD 

```bash
python evals/run_wsd.py acros -- \
  --model_name_or_path <HF_MODEL_OR_LOCAL_CHECKPOINT> \
  --output_json eval_logs/wsd/gloss_activation_target_lemma_colon_raganato_all.json
```

Baselines:

```bash
python evals/run_wsd.py gloss_lm -- \
  --model_name_or_path HuggingFaceTB/SmolLM2-360M \
  --output_json eval_logs/wsd/gloss_lm_base_smollm2_360m_raganato_all.json

python evals/run_wsd.py mfs -- \
  --strategy wordnet_first \
  --output_json eval_logs/wsd/mfs_wordnet_raganato_all.json
```

### 2) CoInCo lexical steering

Build cases:

```bash
python evals/run_steering.py build_coinco -- \
  --output_json eval_logs/coinco_lexsub/coinco_test_cases.json
```

Run target-best ACROS steering:

```bash
python evals/run_steering.py targetbest -- \
  --model_name_or_path <HF_MODEL_OR_LOCAL_CHECKPOINT> \
  --cases_json eval_logs/coinco_lexsub/coinco_test_cases.json \
  --output_json eval_logs/coinco_lexsub/coinco_targetbest.json
```

Run non-oracle self top-k selector:

```bash
python evals/run_steering.py self_topk -- \
  --model_name_or_path <HF_MODEL_OR_LOCAL_CHECKPOINT> \
  --cases_json eval_logs/coinco_lexsub/coinco_test_cases.json \
  --output_json eval_logs/coinco_lexsub/coinco_self_topk.json
```

### 3) SENSiA adaptation + intrinsic evaluation

```bash
python run_adaptation.py \
  --model_name_or_path <HF_MODEL_OR_LOCAL_CHECKPOINT> \
  --output_dir outputs/adapt_eng_ind

python run_post_adaptation_eval.py \
  --model_name_or_path outputs/adapt_eng_ind \
  --output_dir eval_logs/post_adapt

python run_flores_ppl.py \
  --model_name_or_path outputs/adapt_eng_ind \
  --output_json eval_logs/flores_ppl/adapt_eng_ind_sentence.json
```

### 4) XL-Sum generation (paper-style generation metrics)

```bash
python evals/run_generation.py xlsum -- \
  --model_name_or_path <HF_MODEL_OR_LOCAL_CHECKPOINT> \
  --lang ind \
  --split test \
  --output_jsonl eval_logs/xlsum_full/jsonl/acros_ind_fulltest.jsonl

python evals/run_generation.py xlsum_report -- \
  --input_glob 'eval_logs/xlsum_full/jsonl/*.jsonl' \
  --output_json eval_logs/xlsum_full/report.json
```
