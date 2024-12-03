import numpy as np
import pytest

from ramanalysis.peak_fitting import (
    find_n_most_prominent_peaks,
    refine_peak_gaussian_fit,
    refine_peak_parabolic_fit,
)


@pytest.fixture
def get_spiky_sinusoidal():
    """A sum of two sinusoidals with peaks of varying prominence.

    Computed 16 local maxima via WolframAlpha within the range (-2π, 3π):
        x ≈ -5.65189 -> 0.69436
        x ≈ -4.74615 -> 1.09943
        x ≈ -3.84221 -> 0.747582
        x ≈ -2.87123 -> -0.0956415
        x ≈ -1.77297 -> -0.683749
        x ≈ -0.653758 -> -0.382823
        x ≈ 0.366557 -> 0.471284
        x ≈ 1.28262 -> 1.0587
        x ≈ 2.18254 -> 0.919084
        x ≈ 3.12297 -> 0.156366
        x ≈ 4.19206 -> -0.593997
        x ≈ 5.3167 -> -0.557912
        x ≈ 6.37635 -> 0.223944
        x ≈ 7.31039 -> 0.956045
        x ≈ 8.20986 -> 1.03728
        x ≈ 9.12984 -> 0.407268

    See also:
        - https://www.wolframalpha.com/input?i=local+maxima+of+0.9sin%28x%29+%2B+0.2sin%282*pi*x%29+between+-2pi+and+3pi
    """
    x_values = np.linspace(-2 * np.pi, 3 * np.pi, 20000)
    y_values = 0.9 * np.sin(x_values) + 0.2 * np.sin(2 * np.pi * x_values)
    peaks = np.array(
        [
            [-5.65189, 0.69436],
            [-4.74615, 1.09943],
            [-3.84221, 0.747582],
            [-2.87123, -0.0956415],
            [-1.77297, -0.683749],
            [-0.653758, -0.382823],
            [0.366557, 0.471284],
            [1.28262, 1.0587],
            [2.18254, 0.919084],
            [3.12297, 0.156366],
            [4.19206, -0.593997],
            [5.3167, -0.557912],
            [6.37635, 0.223944],
            [7.31039, 0.956045],
            [8.20986, 1.03728],
            [9.12984, 0.407268],
        ]
    )
    x_peaks = peaks[:, 0]
    y_peaks = peaks[:, 1]
    return x_values, y_values, x_peaks, y_peaks


def test_find_2_most_prominent_peaks_on_synthetic_spectrum(valid_synthetic_spectrum_intensities):
    calculated_peak_indices = find_n_most_prominent_peaks(valid_synthetic_spectrum_intensities, 2)
    known_peak_indices = [148, 396]
    np.testing.assert_equal(calculated_peak_indices, known_peak_indices)


def test_find_3_most_prominent_peaks_on_synthetic_spectrum(valid_synthetic_spectrum_intensities):
    calculated_peak_indices = find_n_most_prominent_peaks(valid_synthetic_spectrum_intensities, 3)
    known_peak_indices = [148, 320, 396]
    np.testing.assert_equal(calculated_peak_indices, known_peak_indices)


def test_find_4_most_prominent_peaks_on_synthetic_spectrum(valid_synthetic_spectrum_intensities):
    calculated_peak_indices = find_n_most_prominent_peaks(valid_synthetic_spectrum_intensities, 4)
    known_peak_indices = [29, 148, 320, 396]
    np.testing.assert_equal(calculated_peak_indices, known_peak_indices)


def test_find_5_most_prominent_peaks_on_synthetic_spectrum(valid_synthetic_spectrum_intensities):
    calculated_peak_indices = find_n_most_prominent_peaks(valid_synthetic_spectrum_intensities, 5)
    known_peak_indices = [29, 148, 233, 320, 396]
    np.testing.assert_equal(calculated_peak_indices, known_peak_indices)


def test_find_exactly_most_prominent_peaks_on_spiky_sinusoidal(get_spiky_sinusoidal):
    x_values, y_values, x_peaks, y_peaks = get_spiky_sinusoidal
    calculated_peak_indices = find_n_most_prominent_peaks(y_values, num_peaks=16)
    np.testing.assert_allclose(x_values[calculated_peak_indices], x_peaks, rtol=1e-3)
    np.testing.assert_allclose(y_values[calculated_peak_indices], y_peaks, rtol=1e-3)


def test_find_too_many_prominent_peaks_on_spiky_sinusoidal(get_spiky_sinusoidal):
    x_values, y_values, x_peaks, y_peaks = get_spiky_sinusoidal
    for num_peaks_too_many in [17, 50, 500]:
        calculated_peak_indices = find_n_most_prominent_peaks(
            y_values, num_peaks=num_peaks_too_many
        )
        np.testing.assert_allclose(x_values[calculated_peak_indices], x_peaks, rtol=1e-3)
        np.testing.assert_allclose(y_values[calculated_peak_indices], y_peaks, rtol=1e-3)


def test_find_too_few_prominent_peaks_on_spiky_sinusoidal(get_spiky_sinusoidal):
    x_values, y_values, x_peaks, y_peaks = get_spiky_sinusoidal
    for num_peaks_too_few in [3, 8, 13]:
        calculated_peak_indices = find_n_most_prominent_peaks(y_values, num_peaks=num_peaks_too_few)
        assert calculated_peak_indices.size == num_peaks_too_few


# def test_refine_peak_gaussian_fit(get_spiky_sinusoidal):
#     x_values, y_values, x_peaks, y_peaks = get_spiky_sinusoidal
#     for 


# def test_refine_peak_gaussian_fit():
#     signal = np.array([1, 6, 10, 8, 3])
#     int_peak_index = 2
#     known_subpixel_peak_index = 2.2
#     interpolated_peak_index = refine_peak_gaussian_fit(int_peak_index, signal, window_size=5)
#     np.testing.assert_allclose(known_subpixel_peak_index, interpolated_peak_index, atol=0.1)


# def test_refine_peak_parabolic_fit():
#     signal = np.array([1, 6, 10, 8, 3])
#     int_peak_index = 2
#     known_subpixel_peak_index = 2.2
#     interpolated_peak_index = refine_peak_parabolic_fit(int_peak_index, signal)
#     np.testing.assert_allclose(known_subpixel_peak_index, interpolated_peak_index, atol=0.1)
