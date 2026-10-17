"""Tests for cmapss.models: hand-solved cases, scikit-learn agreement, input validation."""
import numpy as np
import pytest

from cmapss.models import MeanBaseline, RidgeRegressor, boosting_factory


# ---------- ridge: hand-solved cases ----------

def test_ols_recovers_exact_linear_relationship():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 2)) * [1.0, 500.0] + [3.0, 9000.0]     # very different scales
    y = 3 + 2 * X[:, 0] - 0.01 * X[:, 1]
    m = RidgeRegressor(alpha=0).fit(X, y)
    np.testing.assert_allclose(m.coef_, [2, -0.01], atol=1e-8)
    assert m.intercept_ == pytest.approx(3, abs=1e-6)


def test_one_feature_ridge_matches_closed_form_by_hand():
    """x = [-1, 0, 1], y = [1, 2, 3]. Standardised: sd = sqrt(2/3), Z'Z = 3, Z'(y - ybar) = 2/sd.
    So coef = 3 / (3 + alpha) in original units and intercept = ybar = 2."""
    X, y = np.array([[-1.0], [0.0], [1.0]]), np.array([1.0, 2.0, 3.0])
    for alpha in (0.0, 3.0, 9.0):
        m = RidgeRegressor(alpha).fit(X, y)
        assert m.coef_[0] == pytest.approx(3 / (3 + alpha))
        assert m.intercept_ == pytest.approx(2.0)
    assert RidgeRegressor(3.0).fit(X, y).predict([[2.0]])[0] == pytest.approx(3.0)   # 0.5 * 2 + 2


def test_larger_alpha_shrinks_coefficients_towards_zero_and_predictions_to_mean():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(100, 3)); y = X @ [1.0, -2.0, 0.5] + 5 + rng.normal(0, 0.1, 100)
    norms = [np.linalg.norm(RidgeRegressor(a).fit(X, y).coef_) for a in (0, 1, 10, 100, 1e4)]
    assert all(b < a for a, b in zip(norms, norms[1:]))
    huge = RidgeRegressor(1e12).fit(X, y)
    np.testing.assert_allclose(huge.predict(X), y.mean(), atol=1e-4)


def test_predictions_do_not_change_when_a_feature_changes_units():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(80, 3)); y = X @ [1.0, 2.0, 3.0] + rng.normal(0, 0.3, 80)
    X2 = X.copy(); X2[:, 1] *= 1000.0           # e.g. the same quantity expressed in different units
    p1 = RidgeRegressor(5.0).fit(X, y).predict(X)
    p2 = RidgeRegressor(5.0).fit(X2, y).predict(X2)
    np.testing.assert_allclose(p1, p2, atol=1e-8)


def test_constant_column_gets_zero_coefficient_and_does_not_break_fit():
    rng = np.random.default_rng(3)
    X = np.column_stack([rng.normal(size=40), np.full(40, 7.0)])
    y = 2 * X[:, 0] + 1
    m = RidgeRegressor(0.1).fit(X, y)
    assert m.coef_[1] == 0.0 and np.isfinite(m.coef_).all()


def test_perfectly_collinear_columns_still_fit_with_alpha_zero():
    rng = np.random.default_rng(4)
    a = rng.normal(size=60); X = np.column_stack([a, 2 * a]); y = 4 * a + 1
    np.testing.assert_allclose(RidgeRegressor(0).fit(X, y).predict(X), y, atol=1e-8)


# ---------- ridge: agreement with scikit-learn ----------

def test_matches_scikit_learn_standardised_ridge():
    pytest.importorskip("sklearn")
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(5)
    base = rng.normal(size=(300, 3))
    X = np.column_stack([base, base[:, 0] + 0.1 * rng.normal(size=300), base[:, 1] * 100]) + 10   # correlated, mixed scales
    y = X[:, 0] - 2 * X[:, 1] / 100 + rng.normal(0, 0.5, 300)
    X_new = rng.normal(size=(50, 5)) * [1, 1, 1, 1, 100] + 10
    for alpha in (0.5, 2.5, 50.0):
        ours = RidgeRegressor(alpha).fit(X, y).predict(X_new)
        ref = make_pipeline(StandardScaler(), Ridge(alpha=alpha)).fit(X, y).predict(X_new)
        np.testing.assert_allclose(ours, ref, rtol=1e-8, atol=1e-8)


# ---------- validation and state ----------

@pytest.mark.parametrize("model", [RidgeRegressor(1.0), MeanBaseline()])
def test_predict_before_fit_raises(model):
    with pytest.raises(RuntimeError, match="not fitted"):
        model.predict(np.zeros((2, 3)))


def test_negative_alpha_raises():
    with pytest.raises(ValueError, match="alpha"):
        RidgeRegressor(-1)


@pytest.mark.parametrize("model_cls", [RidgeRegressor, MeanBaseline])
def test_fit_rejects_bad_inputs(model_cls):
    good_X, good_y = np.arange(10.0).reshape(5, 2), np.arange(5.0)
    with pytest.raises(ValueError):
        model_cls().fit(good_X[:4], good_y)                 # row mismatch
    with pytest.raises(ValueError):
        model_cls().fit(good_X, good_y.reshape(5, 1))       # y not 1-D
    with pytest.raises(ValueError):
        model_cls().fit(good_X[:1], good_y[:1])             # too few samples
    bad = good_X.copy(); bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN|infinite"):
        model_cls().fit(bad, good_y)


def test_predict_rejects_wrong_width_and_nan():
    m = RidgeRegressor(1.0).fit(np.random.default_rng(0).normal(size=(20, 3)), np.arange(20.0))
    with pytest.raises(ValueError, match="columns"):
        m.predict(np.zeros((4, 2)))
    with pytest.raises(ValueError, match="NaN|infinite"):
        m.predict(np.array([[0.0, np.nan, 0.0]]))


# ---------- mean predictor ----------

def test_mean_baseline_predicts_training_mean_everywhere():
    m = MeanBaseline().fit(np.zeros((4, 2)), np.array([10.0, 20.0, 30.0, 40.0]))
    np.testing.assert_allclose(m.predict(np.zeros((3, 2))), [25.0, 25.0, 25.0])


# ---------- boosting factory ----------

def test_boosting_factory_disables_early_stopping_and_fits():
    pytest.importorskip("sklearn")
    m = boosting_factory(iters=20, leaves=4, min_leaf=5)()
    assert m.early_stopping is False and m.max_iter == 20 and m.max_leaf_nodes == 4
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 3)); y = X[:, 0] * 3
    assert np.corrcoef(m.fit(X, y).predict(X), y)[0, 1] > 0.9
    assert boosting_factory()() is not boosting_factory()()        # a fresh model each call
