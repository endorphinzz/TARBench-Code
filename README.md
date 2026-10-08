# TARBench

TARBench is a benchmark for smart-home trigger-action rule chains, covering both benign long chains and abnormal multi-rule interactions. This open-source package includes the released datasets, data-construction code, and the experimental pipelines for ICL, RAG, and token-compression settings.

## What Is Included

- `data/benign/`
  - `benign_single_rule_chains.json`
  - `benign_long_chains_gpt54.json`
- `data/abnormal/`
  - `abnormal_chains_gpt54.json`
  - `abnormal_rr_re_pie_summary.json`
  - `abnormal_rr_re_radar_summary.json`
- `data/abnormal_fragments/`
  - raw abnormal fragments used to build the released abnormal dataset
- `resources/`
  - `device_ontology.json`
- `generation/`
  - Strategy-B benign long-chain generation script
  - abnormal-chain construction script
  - optional long-chain validation script
- `experiments/`
  - `baselines/` (non-LLM baselines: SMT, graph, heuristic, and learned detectors)
  - `icl/`
  - `rag/`
  - `token_compression/`
  - `datasets/` (mixed evaluation sets)
- `annotation/`
  - annotation guidelines, per-annotator batches, and inter-annotator agreement (IAA) sampling and validation scripts
- `analysis/`
  - surface-cue (artifact) analysis, token-cost analysis, realism comparison, and related figures
- `run.py`
  - unified one-click entry point for common workflows
- `LICENSE` (MIT) and `requirements.txt`

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set API environment variables

The online experiment scripts expect an OpenAI-compatible API endpoint.

```bash
export TARBENCH_API_KEY="YOUR_API_KEY"
export TARBENCH_BASE_URL="https://your-openai-compatible-endpoint/v1"
```

On Windows PowerShell:

```powershell
$env:TARBENCH_API_KEY="YOUR_API_KEY"
$env:TARBENCH_BASE_URL="https://your-openai-compatible-endpoint/v1"
```

### 3. Rebuild reusable artifacts

This regenerates the mixed evaluation sets, the ICL library, and the RAG knowledge base.

```bash
python run.py prepare
```

### 4. Run experiments with one command

Run ICL and automatically evaluate the result:

```bash
python run.py icl --model gpt-5.4 --shot-count 5
```

Run RAG and automatically evaluate the result:

```bash
python run.py rag --model gpt-5.4
```

Run the token-compression benchmark and evaluate it:

```bash
python run.py token-compression --model gpt-5.4
```

Run the full pipeline sequentially:

```bash
python run.py full --model gpt-5.4 --shot-count 5
```

If you only want to verify prompt construction and file generation without calling the API:

```bash
python run.py icl --dry-run
python run.py rag --dry-run
python run.py token-compression --dry-run
```

## Released Results

The repository keeps the evaluation metrics for every reported run under:

- `experiments/icl/outputs/` (`*_metrics.json`)
- `experiments/rag/outputs/` (`*_metrics.json`)
- `experiments/token_compression/outputs/` (`*_metrics.json`)
- `experiments/baselines/outputs/` (`*_metrics.json` and `paper_tables/`)

To keep the package lightweight, the bulky raw per-sample prediction files (which embed the full input rule sets) are not included; all metrics can be reproduced with the commands above. The full dataset and evaluation sets needed for reruns are included under `data/` and `experiments/datasets/`.

## Dataset Construction

### Benign long chains

Strategy B is the main released generator:

```bash
python generation/generate_benign_long_chains.py
```

### Abnormal chains

The released abnormal dataset is built by inserting abnormal fragments into benign long chains:

```bash
python generation/build_abnormal_chains.py
```

### Optional long-chain validation

```bash
python generation/validate_long_chains.py --help
```

## Experiment Layout

### `experiments/icl/`

- build ICL library: `build_icl_library.py`
- run benchmark: `run_icl_benchmark.py`
- evaluate predictions: `evaluate_icl.py`
- compare shot settings: `compare_icl_shots.py`

### `experiments/rag/`

- build knowledge base: `build_knowledge_base.py`
- retrieve context: `retriever.py`
- run benchmark: `run_rag_benchmark.py`
- evaluate predictions: `evaluate_rag.py`

### `experiments/token_compression/`

- build evaluation datasets
- run benchmark
- evaluate predictions

## Notes

- The default public workflow uses environment variables instead of any local private config file.
- The packaged scripts are organized to run from the `TARBench_Code` root directory.
