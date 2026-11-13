from nsh_pipeline.modeling.dataset import LABEL_COLUMN, SPLITS, TRAINING_COLUMNS, load_dataset_card, load_split


def test_every_split_matches_the_contract(training_sample_dir):
    for name in SPLITS:
        assert list(load_split(name, training_sample_dir).columns) == TRAINING_COLUMNS


def test_splits_are_in_time_order_and_disjoint(training_sample_dir):
    train, validation, test = (load_split(name, training_sample_dir) for name in SPLITS)
    assert train["service_date"].max() < validation["service_date"].min()
    assert validation["service_date"].max() < test["service_date"].min()
    ids = [set(df["swipe_id"]) for df in (train, validation, test)]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])


def test_every_split_has_both_classes(training_sample_dir):
    for name in SPLITS:
        assert set(load_split(name, training_sample_dir)[LABEL_COLUMN]) == {0, 1}


def test_card_matches_the_files(training_sample_dir):
    card = load_dataset_card(training_sample_dir)
    for name in SPLITS:
        assert card["splits"][name]["rows"] == len(load_split(name, training_sample_dir))


def test_sample_has_a_learnable_signal(training_sample_dir):
    train = load_split("train", training_sample_dir)
    rate = train.groupby(train["service_category"].isin(["Dental", "Vision"]))[LABEL_COLUMN].mean()
    assert rate[True] > rate[False]  # Dental/Vision denied more often, as designed