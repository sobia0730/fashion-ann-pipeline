# Fashion-MNIST ANN Versioning Report

## Environment

- Python: run `py -3 --version`
- Git: run `git --version`
- DVC: run `dvc --version`
- Target: at least 85% Fashion-MNIST test accuracy

## Git evidence

Capture each command in the terminal for the submission:

```powershell
git log --oneline --graph --all
git log --stat -3
git log -p -1
git log main..dev
git diff
git diff --staged
git diff main..dev
git diff main...dev
git stash list
git log --oneline --graph --all
```

Two-dot compares the two tips directly. Three-dot compares the current branch with the merge base, showing changes introduced on the right-hand branch since divergence.

## DVC evidence

```powershell
dvc dag
dvc repro
dvc status
dvc metrics show
```

Record the DVC console output after the initial run and after changing one value in `params.yaml`. A training hyperparameter change should rerun `train` and `evaluate`; `prepare` and `preprocess` should remain cached.

## Google Drive evidence

After creating a Google Cloud Desktop OAuth client and adding the account as a test user:

```powershell
dvc remote add -d gdriveremote gdrive://YOUR_FOLDER_ID
dvc remote modify gdriveremote gdrive_client_id YOUR_CLIENT_ID
dvc remote modify gdriveremote gdrive_client_secret YOUR_CLIENT_SECRET
dvc push
```

Do not include the client ID/secret or `.dvc/tmp/gdrive-user-credentials.json` in the report or repository.

## Metrics comparison

| Version | Test accuracy | Test loss |
|---|---:|---:|
| v1 | pending local run | pending local run |
| v2 | pending hyperparameter run | pending hyperparameter run |

## Conflict simulation notes

Use two branches that edit the same normalization line and regenerate the processed artifact. Merge the branches, resolve the Python conflict, choose/regenerate the authoritative DVC pointer, run `dvc checkout`, verify `dvc status`, then run `dvc repro`.

Attach screenshots of the conflict markers, resolved file, DVC status, and final graph before exporting this document as PDF.

### Completed simulation

- `teammate-sim` used power normalization: `x ** 0.95`.
- `main` used logarithmic normalization: `log1p(x) / log1p(255)`.
- Merge conflicts occurred in `src/preprocess.py` and `dvc.lock`.
- The main/logarithmic version was selected as authoritative.
- `dvc repro` completed after resolution and `dvc status` was clean.
- Final resolved test accuracy: `0.8794`.
- Final merge commit: `Resolve simulated code and DVC data conflicts`.
