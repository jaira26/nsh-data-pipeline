# Airflow 2.10.5 with the nsh_pipeline package installed in editable mode.
FROM apache/airflow:2.10.5-python3.11

# git: needed by DVC and for the commit hash in run_summary.json
USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && apt-get clean && rm -rf /var/lib/apt/lists/*
USER airflow

COPY --chown=airflow:root pyproject.toml /opt/airflow/project/pyproject.toml
COPY --chown=airflow:root src /opt/airflow/project/src

# Pinning apache-airflow to the image's version stops pip from changing Airflow itself.
# The editable install points at /opt/airflow/project/src, which docker-compose replaces
# with your mounted repo, so code changes apply without rebuilding.
# Replace the dvc range with the exact version you pin in pyproject.toml.
RUN pip install --no-cache-dir \
      "apache-airflow==${AIRFLOW_VERSION}" \
      -e /opt/airflow/project \
      "dvc[gs]>=3.50,<4"

# The repo is mounted with your host user's ownership; let git/DVC operate on it.
RUN git config --global --add safe.directory '*'
