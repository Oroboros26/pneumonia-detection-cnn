# Data Folder

This folder is intended for local dataset management and organization.

## Recommended structure

Place your datasets in a local `data/` folder with the following layout:

```
data/
  train/
    NORMAL/
    PNEUMONIA/
  val/
    NORMAL/
    PNEUMONIA/
  test/
    NORMAL/
    PNEUMONIA/
```

## Usage

- Use `data/dataset_setup.py` to validate paths or download Kaggle datasets if you have credentials configured.
- The notebook currently downloads datasets using `kagglehub`, but local data can be used by updating the notebook paths.

## Notes

- Keep the folder names and subfolder labels consistent with the notebook expectations.
- The CLI tool `predict_cli.py` can be used to run inference on a single image.
