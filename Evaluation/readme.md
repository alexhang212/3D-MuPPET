# Reproducing Benchmarks

All scripts should be run from the **project root directory** (not from `Evaluation/`).

## Prerequisites

- Weights in `Weights/` (download [here](https://doi.org/10.5281/zenodo.19551658))
- Pop3D dataset downloaded, you can download the lighter "EvaluationSequences.zip" for only the sequences used for this evaluation [link](https://github.com/alexhang212/Dataset-3DPOP)
- Output directory `EvaluationData/`

```bash
mkdir -p Data/EvaluationData
```

## Step 1 — Run inference for each model

Run each script independently. They evaluate sequences `[1, 2, 5, 11]` and write results to `Data/EvaluationData/`. Each script skips sequences that already have output files.

```bash
python Evaluation/Eval_YOLODLC.py     --dataset_path /path/to/Pop3D-Dataset
python Evaluation/Eval_YOLOVitPose.py --dataset_path /path/to/Pop3D-Dataset
python Evaluation/Eval_KPRCNN.py      --dataset_path /path/to/Pop3D-Dataset
python Evaluation/Eval_ltohpAlgVol.py --dataset_path /path/to/Pop3D-Dataset
```

## Step 2 — Apply Kalman smoothing

Reads the `SeqEval_Points3D_*.p` files from Step 1 and writes `Kalman3D` pickle files to `Data/EvaluationData/`.

```bash
python Evaluation/KalmanPostProcess.py
```

## Step 3 — Extract metrics

Reads the `Kalman3D` files from Step 2, computes evaluation metrics, and writes summary CSVs to the project root.

```bash
python Evaluation/ExtractMetrics.py --dataset_path /path/to/Pop3D-Dataset
```
