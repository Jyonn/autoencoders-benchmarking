# autoencoders-benchmarking

This project evaluates embedding transforms built with `autoencoders` on downstream tasks.

The initial scope is:

- text classification through MTEB
- a pluggable base encoder layer
- a pluggable embedding transform layer
- a first `autoencoders`-backed transform that can train on task-train embeddings

## Environment

Use the shared research environment:

```bash
source /Users/jyonn/Projects/venv/research/bin/activate
```

Install this project in editable mode:

```bash
pip install -e /Users/jyonn/Projects/Research/autoencoders-benchmarking
```

## First run

```bash
python /Users/jyonn/Projects/Research/autoencoders-benchmarking/scripts/run_classification.py \
  --config /Users/jyonn/Projects/Research/autoencoders-benchmarking/configs/banking77_identity.yaml
```

## Current design

- `base encoder`: produces raw text embeddings
- `embedding transform`: identity or `autoencoders` model
- `task runner`: uses MTEB tasks and evaluation logic

This keeps the benchmark project decoupled from the library itself.
