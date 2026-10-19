import unittest
import numpy as np
from tools.audit_pasa_region_inputs import inspect_segmentation


class RegionInputAuditTests(unittest.TestCase):
    def test_unhandled_label_is_not_silently_remapped(self):
        seg=np.zeros((16,16,9),dtype=np.float32)
        seg[4:8,4:8,3:6]=3
        result=inspect_segmentation(seg,16,7)
        self.assertTrue(result['all_network_regions_empty'])
        self.assertGreater(result['sampled_positive_outside_1_2_4'],0)
        self.assertEqual(result['regions'][2]['network_mask_voxels'],0)

    def test_empty_full_volume_is_distinct(self):
        result=inspect_segmentation(np.zeros((16,16,9)),16,7)
        self.assertTrue(result['all_network_regions_empty'])
        self.assertEqual(result['whole_positive_voxels'],0)

    def test_known_label_preserves_nonempty_region(self):
        seg=np.zeros((16,16,9));seg[4:8,4:8,3:6]=4
        result=inspect_segmentation(seg,16,7)
        self.assertFalse(result['all_network_regions_empty'])
        self.assertEqual(result['sampled_positive_outside_1_2_4'],0)

    def test_nonfinite_segmentation_rejected(self):
        seg=np.zeros((16,16,9));seg[0,0,0]=np.nan
        with self.assertRaises(ValueError):inspect_segmentation(seg,16,7)
