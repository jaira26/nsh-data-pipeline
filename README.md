# nsh-data-pipeline
Data pipeline for synthetic NSH reporting data (Airflow + DVC)

## Data acquisition & preprocessing

```bash
pip install -e ".[dev]"
python -m nsh_pipeline.acquire.download --params params.yaml --out-dir data/raw          # download (or generate) raw data
python -m nsh_pipeline.preprocess.run --raw-dir data/raw --out-dir data/processed --as-of 2025-06-15
pytest --cov=nsh_pipeline
```

- [Data acquisition & synthetic data generation](docs/data_acquisition.md): download stage, generator, `params.yaml`, injected issues
- [Preprocessing](docs/preprocessing.md): cleaning steps and the 9 processed tables
- [Decisions & assumptions](docs/decisions.md)