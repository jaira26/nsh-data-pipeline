"""Model registry, promotion / rollback and pipeline notifications (phase 2, step 5).

    decide.py    which model is the candidate, and whether it may replace production (pure logic)
    bundle.py    package the selected model + reports into one versioned .tar.gz
    backends.py  where versions live: a local folder (tests, PRs) or GCP Artifact Registry
    promote.py   CLI: python -m nsh_pipeline.registry.promote  (DVC stage `register`)
    rollback.py  CLI: python -m nsh_pipeline.registry.rollback --to <version>  (manual rollback)
    notify.py    CLI: python -m nsh_pipeline.registry.notify --event failure  (CI failure alerts)

See docs/model_registry.md.
"""
