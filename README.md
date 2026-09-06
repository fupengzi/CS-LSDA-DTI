[README.md](https://github.com/user-attachments/files/31881200/README.md)
# CS-LSDA-DTI

CS-LSDA-DTI predicts drug-target interactions using MoLFormer, ESM-2 and local sparse dual attention (LSDA). This repository contains the code required for data preparation, feature extraction, model training and deletion-faithfulness analysis.

## Code

```text
model.py       Model architecture, including SDA and LDA
data.py        Data conversion and hot/cold splitting
features.py    MoLFormer and ESM-2 feature extraction
main.py        Training, evaluation and ten-fold experiments
deletion.py    Deletion-faithfulness analysis and plotting
```

## Installation

Python 3.10 and a CUDA GPU are recommended.

```bash
conda create -n cs_lsda_dti python=3.10 -y
conda activate cs_lsda_dti
pip install -r requirements.txt
```

Install the PyTorch build matching your CUDA version when running on a GPU.

## Data and Features

Each input CSV must contain `SMILES`, `Protein` and `Y`. The expected split layout is:

```text
data/<dataset>/{random,e2,e3,e4}/{0..9}/{train,validation,test}.csv
```

Convert the splits and extract pretrained features:

```bash
python data.py convert --input_root data/BioSNAP --output_root data/converted/biosnap
python features.py drug --csv data/converted/biosnap --out_dir embeddings/biosnap/drug
python features.py protein --csv data/converted/biosnap --out_dir embeddings/biosnap/protein
```

The default encoders are MoLFormer-XL and ESM-2-150M. A local Hugging Face model directory may be supplied with `--model_name`.

## Training

Run all hot- and cold-start splits for ten folds:

```bash
python -u main.py run --train --converted_root data/converted/biosnap
```

The dataset name, embedding directories and output directory are inferred from `converted_root`. Use the optional command arguments to override them. Replace `biosnap` with `bindingdb` or `drugbank` for the other datasets.

## Deletion Faithfulness

```bash
python deletion.py run --config outputs/biosnap/random/0/config.yaml --checkpoint outputs/biosnap/random/0/best_model.pt --modality protein --random_repeats 20 --output_dir outputs/deletion/protein
```

Use `--modality drug` for drug-token deletion. Combine protein and drug results with:

```bash
python deletion.py plot --protein_summary outputs/deletion/protein/deletion_summary.csv --drug_summary outputs/deletion/drug/deletion_summary.csv --output outputs/deletion_figure
```

Training uses all configured epochs and saves the checkpoint with the best validation AUROC. 
