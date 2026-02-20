# DeepSEA Transformer-like Baseline (PyTorch)

Minimal runnable baseline aligned to AMBER DeepSEA task:
- input: DNA one-hot sequence `(1000, 4)`
- output: multi-label logits for `919` tasks
- loss: BCEWithLogitsLoss
- metric: macro AUROC across valid labels

## Selectable models via `--model-id`
- `local_transformer`: local `nn.TransformerEncoder` baseline
- `dnabert2`: HuggingFace DNABERT-2 backbone + classification head
- `nucleotide_transformer`: HuggingFace Nucleotide Transformer backbone + classification head

> For `dnabert2` and `nucleotide_transformer`, install:
```bash
pip install transformers
```

## Files
- `dataset.py`: `.mat` loader + `DeepSEADataset` + one-hot→DNA sequence converter
- `model_transformer.py`: model factory (`build_model`) and model definitions
- `train.py`: training entry
- `eval.py`: checkpoint evaluation entry

## Train (local Transformer)
```bash
python examples/DeepSEA/torch_baseline/train.py \
  --train-mat /path/to/train.mat \
  --valid-mat /path/to/valid.mat \
  --model-id local_transformer \
  --outdir ./outputs/torch_transformer \
  --epochs 5 --batch-size 64
```

## Train (DNABERT-2)
```bash
python examples/DeepSEA/torch_baseline/train.py \
  --train-mat /path/to/train.mat \
  --valid-mat /path/to/valid.mat \
  --model-id dnabert2 \
  --outdir ./outputs/dnabert2
```

## Train (Nucleotide Transformer)
```bash
python examples/DeepSEA/torch_baseline/train.py \
  --train-mat /path/to/train.mat \
  --valid-mat /path/to/valid.mat \
  --model-id nucleotide_transformer \
  --outdir ./outputs/nucleotide_transformer
```

## Evaluate
```bash
python examples/DeepSEA/torch_baseline/eval.py \
  --train-mat /path/to/train.mat \
  --valid-mat /path/to/valid.mat \
  --model-id dnabert2 \
  --ckpt ./outputs/dnabert2/best_model.pt
```

## Runtime debugging tips on cluster
- `train.py` and `eval.py` now print timestamped step logs from the main process.
- Default `--num-workers` is set to `0` to avoid silent worker-process failures in some schedulers.
- If stable, you can increase `--num-workers` later.
