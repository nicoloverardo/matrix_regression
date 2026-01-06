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


def test_loading_instance():
    threshold = 0.3
    mr = MatrixRegression(threshold=threshold)

    assert isinstance(mr, MatrixRegression)
    assert mr.threshold == threshold


def test_fit(dummy_data):
    X_train, X_test, y_train, y_test = dummy_data

    mr = MatrixRegression()
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
