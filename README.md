# autoencoders-benchmarking

This project benchmarks `autoencoders`-based embedding transforms on downstream tasks.

The current scope is:

- MTEB text classification
- a pluggable base sentence encoder
- a pluggable embedding transform layer
- reusable benchmark configs split into `data` and `transform`

## Environment

Use the shared research environment:

```bash
source /Users/jyonn/Projects/venv/research/bin/activate
```

Install this project in editable mode:

```bash
pip install -e .
```

## First Run

Identity baseline:

```bash
python scripts/run_classification.py \
  --data config/data/banking77.yaml \
  --transform config/autoencoder/identity.yaml
```

AE with CLI overrides:

```bash
python scripts/run_classification.py \
  --data config/data/banking77.yaml \
  --transform config/autoencoder/ae.yaml \
  --encoder_hidden_dims 256,128,64 \
  --decoder_hidden_dims 64,128,256,384 \
  --latent_dim 64
```

Summarize finished experiment folders into flat tables:

```bash
python scripts/summarize_results.py
```

## Config Layout

```text
config/
  data/
    banking77.yaml
  autoencoder/
    identity.yaml
    ae.yaml
    vae.yaml
    betavae.yaml
    pqvae.yaml
    rqvae.yaml
    rqvae_codes.yaml
    semhash.yaml
```

## Config Style

The benchmark now follows the same broad style as `autoencoders/examples`:

- `--data ...yaml` contains dataset, base encoder, and MTEB task settings
- `--transform ...yaml` contains transform, model, encoder, decoder, and trainer settings
- extra CLI flags are treated as `RefConfig` placeholders

Example placeholder:

```yaml
encoder:
  name: mlp
  config:
    hidden_dims: ${encoder_hidden_dims:256,128,64}$
```

Then:

```bash
python scripts/run_classification.py \
  --data config/data/banking77.yaml \
  --transform config/autoencoder/rqvae.yaml \
  --encoder_hidden_dims 384,192 \
  --num_quantizers 3
```

Sequence-valued fields such as `hidden_dims` and `sinkhorn_epsilon` are written
as comma-separated strings in placeholders and normalized into Python lists at
load time.

## Data Config

`config/data/*.yaml` contains:

- a benchmark data alias such as `banking77-v2`
- the upstream sentence encoder config
- the MTEB task name and output root
- `task.config`, which is forwarded onto the MTEB task instance

For classification tasks, `task.config` is where settings like these live:

- `train_split`
- `input_column_name`
- `label_column_name`
- `samples_per_label`
- `n_experiments`

## Transform Config

`config/autoencoder/*.yaml` follows the `autoencoders` examples layout:

- `model.name` / `model.config`
- `encoder.name` / `encoder.config`
- `decoder.name` / `decoder.config`
- `trainer`

Benchmark-specific fields stay at the transform top level:

- `name`
- `kind`
- `fit`
- `output_representation`
- `batch_size`
- `projection_dim`
- `projection_seed`

If `checkpoint_dir` or `trainer.output_dir` is omitted, the runner fills them in
automatically under:

```text
results/<data-name>-<transform-name>/
```

## Quantized Latent Adapters

For quantized models, the benchmark supports two sequence-to-vector adapters:

- `output_representation: quantized_projected`
  Uses per-slot quantized vectors, concatenates them, then projects back to one
  dense vector.
- `output_representation: code_indices_projected`
  Uses discrete code indices, maps them through random slot-specific embedding
  tables, concatenates them, then projects back to one dense vector.

See:

- `config/autoencoder/rqvae.yaml`
- `config/autoencoder/rqvae_codes.yaml`

## Results Summary

The summarizer scans `results/` and writes:

- `results/_tables/results_summary.csv`
- `results/_tables/results_summary.jsonl`

Each row contains flattened MTEB metrics, base encoder metadata, transform
metadata, resolved benchmark config fields, and autoencoder fit metadata.
