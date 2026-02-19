# DeepSEA Transformer-like Baseline (PyTorch)

Minimal runnable baseline aligned to AMBER DeepSEA task:
- input: DNA one-hot sequence `(1000, 4)`
- output: multi-label logits for `919` tasks
- loss: BCEWithLogitsLoss
- metric: macro AUROC across valid labels

## Files
- `dataset.py`: `.mat` loader + `DeepSEADataset`
- `model_transformer.py`: `DeepSEATransformer`
- `train.py`: training entry
- `eval.py`: checkpoint evaluation entry

## Train
```bash
python examples/DeepSEA/torch_baseline/train.py \
  --train-mat /path/to/train.mat \
  --valid-mat /path/to/valid.mat \
  --outdir ./outputs/torch_transformer \
  --epochs 5 --batch-size 64
```

## Evaluate
```bash
python examples/DeepSEA/torch_baseline/eval.py \
  --train-mat /path/to/train.mat \
  --valid-mat /path/to/valid.mat \
  --ckpt ./outputs/torch_transformer/best_model.pt
```
