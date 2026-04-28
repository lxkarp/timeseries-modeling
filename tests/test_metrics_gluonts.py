import unittest
from types import SimpleNamespace

import numpy as np

from metrics import EMD, PEMD


class Batch(dict):
    pass


class MetricAggregationShapeTest(unittest.TestCase):
    def test_emd_aggregates_with_gluonts_default_axes(self):
        batch = Batch(
            {
                "label": [np.array([1.0, 2.0])],
                "0.5": [np.array([2.0, 1.0])],
                "seasonal_error": [1.0],
            }
        )

        metric = EMD()(axis=(0, 1))
        metric.update(batch)

        self.assertEqual(metric.get(), 0.0)

    def test_pemd_aggregates_with_gluonts_default_axes(self):
        batch = Batch(
            {
                "label": [np.array([0.0, 10.0])],
                "0.5": [np.array([0.0, 10.0])],
                "seasonal_error": [1.0],
            }
        )
        batch.maps = [
            None,
            SimpleNamespace(
                forecasts=[
                    SimpleNamespace(
                        samples=np.array(
                            [
                                [0.0, 10.0],
                                [0.0, 20.0],
                                [10.0, 20.0],
                            ]
                        )
                    )
                ]
            ),
        ]

        metric = PEMD()(axis=(0, 1))
        metric.update(batch)

        self.assertAlmostEqual(metric.get(), 5.0 - (40.0 / 12.0))


if __name__ == "__main__":
    unittest.main()
