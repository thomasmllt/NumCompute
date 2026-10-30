"""Unit tests for the streaming preprocessing (partial_fit of each transformer).

Each transformer is fed chunk by chunk and compared with the same transformer
fitted in one batch, then checked on the edge cases of a stream.
"""
import numpy as np
import pytest

from numcompute_stream.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    SimpleImputer,
    StandardScaler,
)


def _stream(transformer, X, n_chunks):
    for chunk in np.array_split(X, n_chunks):
        transformer.partial_fit(chunk)
    return transformer


def test_standard_scaler_stream_matches_batch():
    X = np.random.default_rng(0).normal(3.0, 2.0, size=(1000, 4))
    stream = _stream(StandardScaler(), X, 7)
    batch = StandardScaler().fit(X)
    assert np.allclose(stream.mean_, X.mean(axis=0))
    assert np.allclose(stream.scale_, X.std(axis=0))
    assert np.allclose(stream.transform(X), batch.transform(X))


def test_standard_scaler_independent_of_chunking():
    X = np.random.default_rng(1).random((300, 3))
    one = _stream(StandardScaler(), X, 1)
    many = _stream(StandardScaler(), X, 300)
    assert np.allclose(one.mean_, many.mean_)
    assert np.allclose(one.scale_, many.scale_)


def test_standard_scaler_ignores_nan_per_feature():
    X = np.array([[1.0, np.nan], [np.nan, 4.0], [3.0, 6.0]])
    scaler = _stream(StandardScaler(), X, 3)
    assert np.allclose(scaler.mean_, [2.0, 5.0])
    assert np.allclose(scaler.scale_, [1.0, 1.0])
    assert np.isnan(scaler.transform(X)[0, 1])


def test_standard_scaler_all_nan_feature():
    X = np.array([[1.0, np.nan], [2.0, np.nan]])
    with np.errstate(all="raise"):
        scaler = _stream(StandardScaler(), X, 2)
    assert np.isnan(scaler.mean_[1])
    assert scaler.scale_[1] == 1.0


def test_standard_scaler_constant_column_with_rounding():
    # 0.1 has no exact binary form: the computed std is not exactly zero.
    X = np.full((90, 2), 0.1)
    scaler = _stream(StandardScaler(), X, 7)
    assert np.allclose(scaler.scale_, 1.0)
    assert np.allclose(scaler.transform(X), 0.0)


def test_standard_scaler_zero_variance_chunk_then_varying():
    scaler = StandardScaler().partial_fit(np.full((5, 1), 2.0))
    scaler.partial_fit([[4.0], [6.0]])
    X = np.array([2.0] * 5 + [4.0, 6.0])
    assert np.isclose(scaler.scale_[0], X.std())


def test_standard_scaler_stable_with_large_offset():
    X = 1e9 + np.random.default_rng(2).random(10000)
    scaler = _stream(StandardScaler(), X, 10)
    assert np.isclose(scaler.scale_[0], np.std(X - 1e9), rtol=1e-6)


def test_standard_scaler_fit_forgets_previous_chunks():
    scaler = StandardScaler().partial_fit([[100.0], [200.0]])
    scaler.fit([[1.0], [3.0]])
    assert np.allclose(scaler.mean_, [2.0])


def test_standard_scaler_1d_chunks():
    scaler = _stream(StandardScaler(), np.arange(10.0), 3)
    assert np.allclose(scaler.mean_, [4.5])
    assert scaler.transform(np.arange(10.0)).shape == (10,)


@pytest.mark.parametrize("make", [StandardScaler, MinMaxScaler, OneHotEncoder,
                                  SimpleImputer])
def test_partial_fit_returns_self(make):
    transformer = make()
    assert transformer.partial_fit([[1.0, 2.0]]) is transformer


@pytest.mark.parametrize("make", [StandardScaler, MinMaxScaler, OneHotEncoder,
                                  SimpleImputer])
def test_empty_first_chunk_leaves_transformer_unfitted(make):
    transformer = make().partial_fit(np.empty((0, 2)))
    with pytest.raises(ValueError):
        transformer.transform([[1.0, 2.0]])


@pytest.mark.parametrize("make", [StandardScaler, MinMaxScaler, OneHotEncoder,
                                  SimpleImputer])
def test_empty_chunk_is_ignored_after_data(make):
    transformer = make().partial_fit([[1.0, 2.0]])
    before = transformer.transform([[1.0, 2.0]])
    transformer.partial_fit(np.empty((0, 2)))
    assert np.array_equal(transformer.transform([[1.0, 2.0]]), before)


@pytest.mark.parametrize("make", [StandardScaler, MinMaxScaler, OneHotEncoder,
                                  SimpleImputer])
def test_partial_fit_rejects_feature_mismatch(make):
    transformer = make().partial_fit(np.ones((3, 2)))
    with pytest.raises(ValueError):
        transformer.partial_fit(np.ones((3, 3)))


@pytest.mark.parametrize("make", [StandardScaler, MinMaxScaler, OneHotEncoder,
                                  SimpleImputer])
def test_partial_fit_rejects_3d(make):
    with pytest.raises(ValueError):
        make().partial_fit(np.ones((2, 2, 2)))


@pytest.mark.parametrize("make", [StandardScaler, MinMaxScaler, SimpleImputer])
def test_partial_fit_rejects_non_numeric(make):
    with pytest.raises(TypeError):
        make().partial_fit(["a", "b"])


def test_minmax_stream_matches_batch():
    X = np.random.default_rng(3).normal(size=(500, 3))
    stream = _stream(MinMaxScaler(feature_range=(-1.0, 1.0)), X, 6)
    batch = MinMaxScaler(feature_range=(-1.0, 1.0)).fit(X)
    assert np.allclose(stream.transform(X), batch.transform(X))


def test_minmax_range_grows_with_new_chunks():
    scaler = MinMaxScaler().partial_fit([[0.0], [1.0]])
    assert np.allclose(scaler.transform([[2.0]]), [[2.0]])
    scaler.partial_fit([[2.0]])
    assert np.allclose(scaler.transform([[2.0]]), [[1.0]])


def test_minmax_constant_column():
    scaler = _stream(MinMaxScaler(), np.full((6, 1), 5.0), 3)
    assert np.allclose(scaler.transform([[5.0]]), [[0.0]])


def test_minmax_ignores_nan_and_keeps_all_nan_feature():
    X = np.array([[1.0, np.nan], [np.nan, np.nan], [3.0, np.nan]])
    with np.errstate(all="raise"):
        scaler = _stream(MinMaxScaler(), X, 3)
    assert scaler.data_min_[0] == 1.0 and scaler.data_max_[0] == 3.0
    assert np.isnan(scaler.data_min_[1])


def test_onehot_expands_categories_and_keeps_columns():
    encoder = OneHotEncoder().partial_fit([[2], [1]])
    first = encoder.transform([[1], [2]])
    assert first.tolist() == [[1.0, 0.0], [0.0, 1.0]]
    encoder.partial_fit([[0], [2]])
    # The new category 0 comes last even though it sorts first.
    assert encoder.categories_[0].tolist() == [1, 2, 0]
    assert np.array_equal(encoder.transform([[1], [2]])[:, :2], first)


def test_onehot_stream_has_same_categories_as_batch():
    X = np.random.default_rng(4).integers(0, 6, size=(200, 2))
    stream = _stream(OneHotEncoder(), X, 9)
    batch = OneHotEncoder().fit(X)
    for found, expected in zip(stream.categories_, batch.categories_):
        assert sorted(found.tolist()) == expected.tolist()
    assert np.array_equal(stream.transform(X).sum(axis=1), np.full(200, 2.0))


def test_onehot_strings_and_longer_new_category():
    encoder = OneHotEncoder().partial_fit(["a", "b"])
    encoder.partial_fit(["c_long", "a"])
    assert encoder.categories_[0].tolist() == ["a", "b", "c_long"]
    assert encoder.transform(["c_long"]).tolist() == [[0.0, 0.0, 1.0]]


def test_onehot_nan_is_not_a_category():
    X = np.array([[1.0], [np.nan]])
    encoder = _stream(OneHotEncoder(), X, 2).partial_fit(X)
    assert encoder.categories_[0].tolist() == [1.0]
    assert encoder.transform([[np.nan]]).tolist() == [[0.0]]


def test_onehot_unseen_category_is_all_zeros():
    encoder = OneHotEncoder().partial_fit([[1], [2]])
    assert encoder.transform([[3]]).tolist() == [[0.0, 0.0]]


def test_imputer_mean_stream_matches_batch():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(400, 3))
    X[rng.random(X.shape) < 0.2] = np.nan
    imputer = _stream(SimpleImputer(strategy="mean"), X, 8)
    assert np.allclose(imputer.statistics_, np.nanmean(X, axis=0))


def test_imputer_estimate_changes_with_new_chunks():
    imputer = SimpleImputer(strategy="mean").partial_fit([[2.0]])
    assert imputer.transform([[np.nan]]).tolist() == [[2.0]]
    imputer.partial_fit([[4.0]])
    assert imputer.transform([[np.nan]]).tolist() == [[3.0]]


def test_imputer_median_exact_below_capacity():
    rng = np.random.default_rng(6)
    X = rng.normal(size=(301, 2))
    X[rng.random(X.shape) < 0.1] = np.nan
    imputer = _stream(SimpleImputer(strategy="median", max_samples=500), X, 5)
    assert np.allclose(imputer.statistics_, np.nanmedian(X, axis=0))


def test_imputer_median_approximate_above_capacity():
    X = np.random.default_rng(7).random((50000, 1))
    imputer = SimpleImputer(strategy="median", max_samples=2000, seed=0)
    _stream(imputer, X, 25)
    assert abs(imputer.statistics_[0] - 0.5) < 0.05


def test_imputer_constant_strategy():
    imputer = _stream(SimpleImputer(strategy="constant", fill_value=-1.0),
                      np.ones((4, 2)), 2)
    assert imputer.transform([[np.nan, 2.0]]).tolist() == [[-1.0, 2.0]]


def test_imputer_constant_requires_fill_value():
    with pytest.raises(ValueError):
        SimpleImputer(strategy="constant").partial_fit([[1.0]])


def test_imputer_all_nan_feature_stays_nan():
    X = np.array([[1.0, np.nan], [2.0, np.nan]])
    with np.errstate(all="raise"):
        imputer = _stream(SimpleImputer(), X, 2)
    assert imputer.statistics_[0] == 1.5
    assert np.isnan(imputer.transform(X)[0, 1])


def test_imputer_transform_1d_and_keeps_input_unchanged():
    X = np.array([1.0, np.nan, 3.0])
    imputer = SimpleImputer().partial_fit(X)
    assert imputer.transform(X).tolist() == [1.0, 2.0, 3.0]
    assert np.isnan(X[1])


def test_imputer_fit_forgets_previous_chunks():
    imputer = SimpleImputer().partial_fit([[100.0]])
    imputer.fit([[1.0], [3.0]])
    assert np.allclose(imputer.statistics_, [2.0])


def test_imputer_invalid_strategy_message():
    with pytest.raises(ValueError, match="mode"):
        SimpleImputer(strategy="mode")
