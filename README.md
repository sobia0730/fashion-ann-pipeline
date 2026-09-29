# Fashion-MNIST ANN Pipeline

End-to-end TensorFlow ANN classification pipeline managed with Git and DVC.

## Quick start

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python src/prepare.py
python src/preprocess.py
python src/train.py
python src/evaluate.py
```

The reproducible pipeline is run with `dvc repro` after DVC is installed and initialized.

## DVC Google Drive setup

```powershell
pip install "dvc[gdrive]"
dvc init
dvc remote add -d gdriveremote gdrive://YOUR_FOLDER_ID
git add .dvc/config
# Configure a GCP Desktop OAuth client without committing secrets:
dvc remote modify gdriveremote gdrive_client_id YOUR_CLIENT_ID
dvc remote modify gdriveremote gdrive_client_secret YOUR_CLIENT_SECRET
dvc push
```

Use a Google Cloud OAuth app in Testing mode with your account listed as a test user. Never commit `.dvc/tmp`, client secrets, or local credentials.

## Pipeline stages

1. `prepare.py` downloads Fashion-MNIST and writes `data/raw/data.npz`.
2. `preprocess.py` normalizes and splits the data into train/validation/test arrays.
3. `train.py` trains a fully-connected ANN and writes the model and history.
4. `evaluate.py` writes `metrics.json` and a confusion matrix image.
