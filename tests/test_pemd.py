import unittest

from pemd import (
    probabilistic_value_distribution_emd,
    value_distribution_emd,
)


class ValueDistributionEMDTest(unittest.TestCase):
    def test_value_distribution_emd_ignores_horizon_order(self):
        realized = [3.0, 1.0, 2.0]
        shifted_forecast = [2.0, 3.0, 1.0]

        self.assertEqual(value_distribution_emd(realized, shifted_forecast), 0.0)

    def test_value_distribution_emd_averages_sorted_absolute_differences(self):
        realized = [0.0, 10.0]
        forecast = [2.0, 4.0]

        self.assertEqual(value_distribution_emd(realized, forecast), 4.0)


class ProbabilisticValueDistributionEMDTest(unittest.TestCase):
    def test_pemd_reduces_to_emd_for_degenerate_forecast_distribution(self):
        realized = [0.0, 10.0]
        samples = [[2.0, 4.0]]

        self.assertEqual(
            probabilistic_value_distribution_emd(realized, samples),
            value_distribution_emd(realized, samples[0]),
        )

    def test_pemd_uses_fair_ensemble_spread_correction(self):
        realized = [0.0, 10.0]
        samples = [
            [0.0, 10.0],
            [0.0, 20.0],
            [10.0, 20.0],
        ]

        # First term: mean d_vd(sample, realized) = (0 + 5 + 10) / 3 = 5.
        # Fair spread term: one half of the mean over ordered non-self pairs.
        # Pairwise ordered distances are 5, 10, 5, 5, 10, 5, so correction = 40 / 12.
        self.assertAlmostEqual(
            probabilistic_value_distribution_emd(realized, samples),
            5.0 - (40.0 / 12.0),
        )

    def test_pemd_rejects_empty_forecast_ensembles(self):
        with self.assertRaisesRegex(ValueError, "at least one"):
            probabilistic_value_distribution_emd([1.0], [])


if __name__ == "__main__":
    unittest.main()
