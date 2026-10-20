import unittest

import numpy as np
import torch

from pasa_region_inputs import (EmptyRegionInputError, lesion_slab_plan,
                                select_region_inputs, validate_segmentation_labels)


class RegionInputTests(unittest.TestCase):
    def test_separated_clusters_select_largest_slice_instead_of_empty_midpoint(self):
        seg = np.zeros((10, 10, 30), dtype=np.uint8)
        seg[1:3, 1:3, 2] = 1
        seg[1:6, 1:6, 25] = 3
        self.assertEqual(int(seg[:, :, 14].sum()), 0)
        result = select_region_inputs(seg, roi_size=10, margin=0)
        self.assertEqual(result['z_center'], 25)
        self.assertEqual(result['z_indices'], list(range(22, 29)))
        self.assertTrue(bool(result['region_present'][2]))
        self.assertGreater(float(result['whole_lesion_mask'].sum()), 0)

    def test_ties_choose_lowest_z_and_boundary_clipping_is_explicit(self):
        seg = np.zeros((4, 4, 9), dtype=np.uint8)
        seg[1:3, 1:3, 0] = 1
        seg[1:3, 1:3, 8] = 3
        plan = lesion_slab_plan(seg)
        self.assertEqual(plan['z_center'], 0)
        self.assertEqual(plan['z_indices'], [0, 0, 0, 0, 1, 2, 3])
        self.assertEqual(plan['audit']['z_center_tie_rule'], 'lowest_z_index')

    def test_label_three_stays_separate_from_four(self):
        seg = np.zeros((5, 5, 7), dtype=np.float32)
        seg[1:3, 1:3, 3] = 3
        seg[3, 3, 3] = 4
        result = select_region_inputs(seg, roi_size=5, margin=8)
        self.assertEqual(result['region_masks'].shape, (4, 7, 5, 5))
        self.assertEqual(result['region_present'].tolist(), [False, False, True, True])
        self.assertEqual(result['audit']['slab_voxel_counts_before_resize']['3'], 4)
        self.assertTrue(torch.equal(result['whole_lesion_mask'], result['region_masks'].amax(0)))

    def test_unknown_five_rejected_unless_explicitly_accepted_as_separate_bin(self):
        seg = np.zeros((3, 3, 7), dtype=np.uint8)
        seg[1, 1, 3] = 5
        with self.assertRaisesRegex(ValueError, 'Unaccepted raw segmentation labels'):
            select_region_inputs(seg)
        result = select_region_inputs(seg, region_labels=(1, 2, 3, 4, 5), roi_size=3)
        self.assertEqual(result['region_present'].tolist(), [False, False, False, False, True])
        self.assertEqual(result['audit']['raw_label_counts']['5'], 1)

    def test_all_zero_is_rejected_without_substitution(self):
        with self.assertRaisesRegex(EmptyRegionInputError, 'No accepted-label voxels'):
            select_region_inputs(np.zeros((4, 4, 7), dtype=np.uint8))

    def test_small_bin_lost_in_resize_becomes_absent_and_is_audited(self):
        seg = np.zeros((8, 8, 7), dtype=np.uint8)
        seg[:4, :4, 3] = 1
        seg[7, 7, 3] = 3
        result = select_region_inputs(seg, roi_size=2, margin=0)
        self.assertEqual(result['region_present'].tolist(), [True, False, False, False])
        self.assertEqual(result['audit']['labels_lost_at_resize'], [3])
        self.assertEqual(result['audit']['slab_voxel_counts_after_resize']['3'], 0)

    def test_all_bins_lost_in_resize_is_explicit_error(self):
        seg = np.zeros((8, 8, 7), dtype=np.uint8)
        seg[1, 1, 3] = 1
        with self.assertRaisesRegex(EmptyRegionInputError, 'disappeared during nearest resize'):
            select_region_inputs(seg, roi_size=1, margin=8)

    def test_invalid_segmentation_values_are_not_rounded_or_relabelled(self):
        for value in (1.5, np.nan, np.inf, -1, 6):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_segmentation_labels(np.full((2, 2, 2), value))
        with self.assertRaises(ValueError):
            validate_segmentation_labels(np.zeros((2, 2, 2)), region_labels=(1, 1))


if __name__ == '__main__':
    unittest.main()
