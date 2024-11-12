import numpy as np

from ramanalysis.peak_fitting import (
    find_n_most_prominent_peaks,
    refine_peak_gaussian_fit,
    refine_peak_parabolic_fit,
)


def test_find_2_most_prominent_peaks_on_synthetic_spectrum(valid_synthetic_spectrum_intensities):
    """"""
    calculated_peak_indices = find_n_most_prominent_peaks(valid_synthetic_spectrum_intensities, 2)
    known_peak_indices = [148, 396]
    np.testing.assert_equal(calculated_peak_indices, known_peak_indices)


def test_find_3_most_prominent_peaks_on_synthetic_spectrum(valid_synthetic_spectrum_intensities):
    """"""
    calculated_peak_indices = find_n_most_prominent_peaks(valid_synthetic_spectrum_intensities, 3)
    known_peak_indices = [148, 320, 396]
    np.testing.assert_equal(calculated_peak_indices, known_peak_indices)


def test_find_4_most_prominent_peaks_on_synthetic_spectrum(valid_synthetic_spectrum_intensities):
    """"""
    calculated_peak_indices = find_n_most_prominent_peaks(valid_synthetic_spectrum_intensities, 4)
    known_peak_indices = [29, 148, 320, 396]
    np.testing.assert_equal(calculated_peak_indices, known_peak_indices)


def test_find_5_most_prominent_peaks_on_synthetic_spectrum(valid_synthetic_spectrum_intensities):
    """"""
    calculated_peak_indices = find_n_most_prominent_peaks(valid_synthetic_spectrum_intensities, 5)
    known_peak_indices = [29, 148, 233, 320, 396]
    np.testing.assert_equal(calculated_peak_indices, known_peak_indices)


def test_refine_peak_gaussian_fit():
    signal = np.array([1, 6, 10, 8, 3])
    int_peak_index = 2
    known_subpixel_peak_index = 2.2
    interpolated_peak_index = refine_peak_gaussian_fit(int_peak_index, signal, window_size=5)
    np.testing.assert_allclose(known_subpixel_peak_index, interpolated_peak_index, atol=0.1)


def test_refine_peak_parabolic_fit():
    signal = np.array([1, 6, 10, 8, 3])
    int_peak_index = 2
    known_subpixel_peak_index = 2.2
    interpolated_peak_index = refine_peak_parabolic_fit(int_peak_index, signal)
    np.testing.assert_allclose(known_subpixel_peak_index, interpolated_peak_index, atol=0.1)
