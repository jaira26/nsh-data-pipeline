# nsh-data-pipeline

[![CI](https://github.com/jaira26/nsh-data-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/jaira26/nsh-data-pipeline/actions/workflows/ci.yml)

Data pipeline for synthetic NSH reporting data (Airflow + DVC)

## Data acquisition & preprocessing

```bash
pip install -r requirements.txt    # exact versions CI tests with
pip install -e ".[dev]"
python -m nsh_pipeline.acquire.download --params params.yaml --out-dir data/raw          # download (or generate) raw data
python -m nsh_pipeline.preprocess.run --raw-dir data/raw --out-dir data/processed --as-of 2025-06-15
pytest --cov=nsh_pipeline
```

- [Data acquisition & synthetic data generation](docs/data_acquisition.md): download stage, generator, `params.yaml`, injected issues
- [Preprocessing](docs/preprocessing.md): cleaning steps and the 9 processed tables
- [Bias detection & mitigation](docs/bias.md): data slicing, flag rules, reweighing, evidence on clean and injected data
- [Decisions & assumptions](docs/decisions.md)