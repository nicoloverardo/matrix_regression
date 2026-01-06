"""Main unit test module."""

import multiprocessing

import numpy as np
import pytest
from sklearn.model_selection import train_test_split

from matrixreg.matrixregression import MatrixRegression


@pytest.fixture(scope="session", name="dummy_data")
def get_dummy_data():
    X = np.array(
        [
            "lorem ipsum dolor sit amet consectetur adipiscing elit",
            "suspendisse pellentesque laoreet ligula",
            "sed volutpat ligula elementum mattis aliquet",
            "sed condimentum tempus porttitor",
            "interdum et malesuada fames ac ante ipsum primis in faucibus",
            "suspendisse semper pulvinar lectus vel imperdiet ipsum",
            "curabitur ultricies dapibus elit a eleifend",
            "curabitur molestie ante a malesuada imperdiet",
            "suspendisse vitae molestie enim a malesuada augue",
            "praesent vestibulum ligula vitae lacinia convallis",
        ]
    )

    y = np.zeros((10, 3), dtype=int)
    y[0, [1, 2]] = 1
    y[1, [0, 1, 2]] = 1
    y[2, 1] = 1
    y[4, [0, 2]] = 1
    y[6, [0, 1, 2]] = 1
    y[7, [0, 2]] = 1
    y[9, 2] = 1

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test


@pytest.fixture
def dummy_data_extended(dummy_data):
    """Extended dummy data with additional categories for partial_fit testing."""
    X_train, X_test, y_train, y_test = dummy_data
    # Add a new category
    y_extended = np.zeros((y_train.shape[0], y_train.shape[1] + 1), dtype=int)
    y_extended[:, :-1] = y_train
    y_extended[0, -1] = 1  # Add some labels to new category
    return X_train, y_extended


def test_loading_instance():
    threshold = 0.3
    mr = MatrixRegression(threshold=threshold)

    assert isinstance(mr, MatrixRegression)
    assert mr.threshold == threshold


def test_threshold(dummy_data):
    threshold = -2
    mr = MatrixRegression(threshold=threshold)

    X_train, X_test, y_train, y_test = dummy_data
    with pytest.raises(ValueError):
        mr.fit(X_train, y_train)

    assert isinstance(mr, MatrixRegression)
    assert mr.threshold == threshold


@pytest.mark.parametrize("threshold", [0.3, None])
def test_fit(dummy_data, threshold):
    X_train, X_test, y_train, y_test = dummy_data

    mr = MatrixRegression(threshold=threshold)
    mr.fit(X_train, y_train)

    old_shape = mr.W.shape + tuple()

    assert mr.W.shape != (0, 0)

    mr.partial_fit(X_test, y_test)

    assert mr.W.shape != old_shape


def test_predict(dummy_data):
    X_train, X_test, y_train, y_test = dummy_data

    mr = MatrixRegression()
    mr.fit(X_train, y_train)

    yhat = mr.predict(X_test)

    assert yhat.shape == y_test.shape


def test_n_jobs_none():
    """Test n_jobs set to 1 when None."""
    mr = MatrixRegression(n_jobs=None)
    X = ["test document"]
    y = np.array([[1]])
    mr.fit(X, y)
    assert mr.n_jobs == 1


def test_n_jobs_zero():
    """Test n_jobs set to 1 when 0."""
    mr = MatrixRegression(n_jobs=0)
    X = ["test document"]
    y = np.array([[1]])
    mr.fit(X, y)
    assert mr.n_jobs == 1


def test_n_jobs_minus_one():
    """Test n_jobs set to cpu_count when -1."""

    mr = MatrixRegression(n_jobs=-1)
    X = ["test document"]
    y = np.array([[1]])
    mr.fit(X, y)
    assert mr.n_jobs == multiprocessing.cpu_count()


def test_n_jobs_large():
    """Test n_jobs set to cpu_count when larger than available."""

    mr = MatrixRegression(n_jobs=multiprocessing.cpu_count() + 10)
    X = ["test document"]
    y = np.array([[1]])
    mr.fit(X, y)
    assert mr.n_jobs == multiprocessing.cpu_count()


def test_partial_fit_new_categories(dummy_data, dummy_data_extended):
    """Test partial_fit with new categories."""
    X_train, _, y_train, _ = dummy_data
    _, y_extended = dummy_data_extended
    mr = MatrixRegression()
    mr.fit(X_train, y_train)
    old_n_categories = mr.W.shape[1]
    mr.partial_fit(X_train, y_extended)
    assert mr.W.shape[1] > old_n_categories


def test_predict_with_list_X(dummy_data):
    """Test predict with X as list."""
    X_train, X_test, y_train, _ = dummy_data
    mr = MatrixRegression()
    mr.fit(X_train, y_train)
    X_list = X_test.tolist()
    yhat = mr.predict(X_list)
    assert yhat.shape[0] == len(X_list)


def test_n_jobs_valid():
    """Test n_jobs set to given value when valid."""
    valid_n = min(2, multiprocessing.cpu_count())
    mr = MatrixRegression(n_jobs=valid_n)
    X = ["test document"]
    y = np.array([[1]])
    mr.fit(X, y)
    assert mr.n_jobs == valid_n


def test_get_number_categories_invalid():
    """Test _get_number_categories with invalid y."""
    mr = MatrixRegression()
    with pytest.raises(ValueError, match="Cannot get the number of categories"):
        mr._get_number_catgories("invalid")
