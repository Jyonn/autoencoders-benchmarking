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
pip install -e .
```

## First run

```bash
python scripts/run_classification.py --config configs/banking77_identity.yaml
```

## Configuring Autoencoders Models

For `transform.kind: autoencoder`, this project deliberately keeps the
`autoencoders` configuration surface open.

- `transform.model_name` selects the model family such as `ae`, `vae`, `rqvae`
- `transform.model_config` is forwarded directly to the model config class
- `transform.encoder_name` / `transform.encoder_config` configure the encoder backbone
- `transform.decoder_name` / `transform.decoder_config` configure the decoder backbone
- `transform.training_config` is forwarded directly to the matching trainer config

That means simple settings like `latent_dim` and `hidden_dims`, as well as
family-specific settings like `num_quantizers`, `num_codebooks`,
`assignment_strategy`, `sinkhorn_epsilon`, `sinkhorn_iters`, or `codebook_size`
can all be expressed directly in YAML.

Example:

```yaml
transform:
  kind: autoencoder
  model_name: rqvae
  output_representation: latents
  model_config:
    latent_dim: 128
    num_quantizers: 4
    codebook_size: 256
    assignment_strategy: sinkhorn
    sinkhorn_epsilon: [0.05, 0.05, 0.05, 0.05]
    sinkhorn_iters: 50
  encoder_name: mlp
  encoder_config:
    hidden_dims: [384, 256]
    activation: relu
  decoder_name: mlp
  decoder_config:
    hidden_dims: [256, 384]
    activation: relu
  training_config:
    epochs: 0
    patience: 5
    batch_size: 256
    learning_rate: 0.001
```

See:

- `configs/banking77_ae.yaml`
- `configs/banking77_rqvae.yaml`

## Current design

- `base encoder`: produces raw text embeddings
- `embedding transform`: identity or `autoencoders` model
- `task runner`: uses MTEB tasks and evaluation logic

This keeps the benchmark project decoupled from the library itself.
