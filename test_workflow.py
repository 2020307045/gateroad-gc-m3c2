"""Numerical checks with in-memory inputs; no manuscript data are distributed."""
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

os.environ.setdefault('MPLBACKEND', 'Agg')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import numpy as np
import pandas as pd
import generate_gc_m3c2_deformation_cloudmap as gc
from reproduce_gateroad_denoising import detect_attached_noise, radius_density_filter
from evaluate_monitoring_accuracy import evaluate
from summarize_deformation import summarize


class ProjectionTests(unittest.TestCase):
    def setUp(self):
        x = np.linspace(0, 2, 21)
        self.base = np.column_stack([x, np.zeros_like(x), np.full_like(x, 2.0)])
        self.centers = self.base.copy()
        self.centers[:, 2] = 0
        self.args = SimpleNamespace(projection_radius=0.25, half_length=0.5,
                                    min_neighbors=2, voxel_size=0, peak_alpha=0.4,
                                    peak_percentile=95, section_step=1)

    def compute(self, target):
        return gc.compute_gc_m3c2(self.base, target, np.array([1.,0,0]),
                                  self.centers, np.zeros(3), 0, self.args)

    def test_identical_epochs_are_zero(self):
        np.testing.assert_allclose(self.compute(self.base)[0], 0, atol=1e-12)

    def test_inward_and_outward_sign(self):
        for translation in [-0.1, 0.1]:
            target = self.base.copy()
            target[:, 2] += translation
            np.testing.assert_allclose(self.compute(target)[0], -translation, atol=1e-12)

    def test_missing_neighborhood_is_nan(self):
        result, _, counts, _, _ = self.compute(self.base + [0, 0, 5])
        self.assertTrue(np.isnan(result).all())
        self.assertTrue((counts == 0).all())

    def test_peak_blend_on_known_projection_distribution(self):
        base = np.tile([0., 0., 2.], (5, 1))
        target = base.copy()
        target[:, 2] -= [0, 0, 0, 0, 0.4]
        self.args.peak_percentile = 95
        values = gc.compute_gc_m3c2(base, target, np.array([1.,0,0]),
            np.array([[-1.,0,0],[0.,0,0],[1.,0,0]]), np.zeros(3), 0, self.args)[0]
        np.testing.assert_allclose(values, 0.128, atol=1e-12)


class DenoisingTests(unittest.TestCase):
    def test_curvature_alone_does_not_remove_points(self):
        mask, _, _ = detect_attached_noise(np.array([0.,0.,0.,1.]), np.ones(4), 0.5, 0)
        self.assertFalse(mask.any())

    def test_attached_requires_both_constraints(self):
        mask, _, _ = detect_attached_noise(np.array([0.,0.,0.,1.]), np.array([1.,1.,1.,10.]), 0.5, 0)
        np.testing.assert_array_equal(mask, [False,False,False,True])

    def test_radius_excludes_self(self):
        mask, count = radius_density_filter(np.array([[0.,0,0],[0.1,0,0],[10.,0,0]]), 0.2, 1)
        np.testing.assert_array_equal(count, [1,1,0])
        np.testing.assert_array_equal(mask, [False,False,True])


class ReportingTests(unittest.TestCase):
    def test_valid_weighted_statistics_and_boundary(self):
        frame = pd.DataFrame({'valid_flag':[1,1,1,0], 'mileage_m':[0,2,20,5],
            'signed_gc_m3c2_distance':[0.1,-0.1,0.3,999]})
        intervals, stats = summarize(frame, 10, 850)
        self.assertEqual(len(intervals), 2)
        self.assertEqual(intervals.valid_count.tolist(), [2,1])
        self.assertAlmostEqual(stats['mean_mm'], 100)
        self.assertEqual(stats['min_mm'], -100)
        self.assertEqual(stats['valid_ratio'], 0.75)
        self.assertAlmostEqual(np.average(intervals.mean_mm, weights=intervals.valid_count), stats['mean_mm'])

    def test_empty_interval_preserved(self):
        frame = pd.DataFrame({'valid_flag':[1,1], 'mileage_m':[0,25], 'signed_gc_m3c2_distance':[0.1,0.2]})
        intervals, _ = summarize(frame)
        self.assertEqual(intervals.valid_count.tolist(), [1,0,1])
        self.assertTrue(np.isnan(intervals.mean_mm.iloc[1]))

    def test_invalid_statistics_input(self):
        frame = pd.DataFrame({'valid_flag':[1], 'mileage_m':[0], 'signed_gc_m3c2_distance':[np.nan]})
        with self.assertRaises(ValueError):
            summarize(frame)

    def measurement_frame(self):
        return pd.DataFrame({'chainage_m':[0,1,0,1], 'location':['roof']*4,
            'method':['A','A','B','B'], 'measured_mm':[0,100,0,100], 'estimated_mm':[10,80,0,90]})

    def test_accuracy_and_zero_reference(self):
        detailed, summary = evaluate(self.measurement_frame())
        self.assertTrue(np.isnan(detailed.relative_error_pct.iloc[0]))
        self.assertEqual(summary.mae_mm.tolist(), [15,5])
        self.assertEqual(summary.bias_mm.tolist(), [-5,-5])

    def test_unmatched_methods_rejected(self):
        with self.assertRaises(ValueError):
            evaluate(self.measurement_frame().iloc[:-1])

    def test_conflicting_references_rejected(self):
        frame = self.measurement_frame()
        frame.loc[3, 'measured_mm'] = 99
        with self.assertRaises(ValueError):
            evaluate(frame)


if __name__ == '__main__':
    unittest.main()
