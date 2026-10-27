import json

import numpy as np
import pandas as pd

from nsh_pipeline.validation.stats import (
    column_kind, column_stats, compute_stats, read_stats, table_stats, write_stats,
)


def test_column_kinds_do_not_depend_on_pandas_text_dtype():
    assert column_kind(pd.Series(["a"], dtype="str")) == "text"
    assert column_kind(pd.Series(["a"], dtype=object)) == "text"
    assert column_kind(pd.Series([1.5])) == "number"
    assert column_kind(pd.Series([True])) == "bool"
    assert column_kind(pd.Series(pd.to_datetime(["2025-01-01"], utc=True))) == "datetime"


def test_numeric_stats_and_null_rate():
    out = column_stats(pd.Series([1.0, 2.0, 3.0, np.nan]))
    assert out == {"kind": "number", "null_count": 1, "null_rate": 0.25, "min": 1.0, "max": 3.0,
                   "mean": 2.0, "std": 1.0, "median": 2.0}


def test_all_null_and_single_value_columns_give_none_not_nan():
    assert column_stats(pd.Series([np.nan, np.nan]))["mean"] is None
    assert column_stats(pd.Series([5.0]))["std"] is None
    assert column_stats(pd.Series([], dtype=float))["null_rate"] is None


def test_text_top_k_is_sorted_with_ties_broken_by_value():
    out = column_stats(pd.Series(["b", "a", "b", "c", "a", None]), top_k=2)
    assert out["distinct"] == 3
    assert out["top"] == [{"value": "a", "count": 2, "share": 0.333333},
                          {"value": "b", "count": 2, "share": 0.333333}]
    assert out["top_complete"] is False


def test_high_cardinality_text_has_no_top_k():
    out = column_stats(pd.Series([f"id{i}" for i in range(60)]), max_categories=50)
    assert out["distinct"] == 60 and "top" not in out


def test_bool_and_datetime_stats():
    assert column_stats(pd.Series([True, False, True, True]))["true_rate"] == 0.75
    dates = column_stats(pd.Series(pd.to_datetime(["2025-01-02", "2025-03-04", None], utc=True)))
    assert dates["min"] == "2025-01-02T00:00:00+00:00" and dates["max"] == "2025-03-04T00:00:00+00:00"


def test_table_stats_cover_every_column(clean_tables, settings):
    stats = compute_stats(clean_tables, settings)
    assert sorted(stats) == sorted(clean_tables)
    swipes = stats["swipes"]
    assert swipes["row_count"] == len(clean_tables["swipes"])
    assert list(swipes["columns"]) == list(clean_tables["swipes"].columns)
    assert swipes["columns"]["status"]["kind"] == "text"


def test_stats_are_identical_for_object_and_str_text(clean_tables):
    df = clean_tables["tickets"]
    as_object = df.astype({c: object for c in df.columns if isinstance(df[c].dtype, pd.StringDtype)})
    assert table_stats(df, "tickets") == table_stats(as_object, "tickets")


def test_write_and_read_round_trip(clean_tables, settings, tmp_path):
    stats = compute_stats(clean_tables, settings)
    paths = write_stats(stats, tmp_path / "stats")
    assert sorted(p.name for p in paths) == sorted(f"{n}.json" for n in clean_tables)
    assert read_stats(tmp_path / "stats") == json.loads(json.dumps(stats))
    assert read_stats(tmp_path / "missing") == {}
