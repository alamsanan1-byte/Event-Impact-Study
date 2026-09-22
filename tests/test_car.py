import numpy as np


def test_car_equals_sum_of_ars():
    ar = np.array([0.01, -0.02, 0.005])
    car = ar.sum()
    assert np.isclose(car, -0.005)
