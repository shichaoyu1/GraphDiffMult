"""Synthetic checks of input masks, loss supervision and validation RNG isolation."""
import copy
import random
import unittest

import numpy as np
import torch

from pasa_repair_pilot import (METHODS, RepairAlignmentModel, fieldwise_loss,
                               masked_mean, validation_rng)
from recent_alignment import masked_nce


class RepairPilotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def model(self, method='region_pool_field', graph=False):
        torch.manual_seed(73)
        return RepairAlignmentModel(method, num_anchors=4, z_slices=3,
                                    feat_dim=16, shared_dim=8, graph=graph)

    def inputs(self, regions=5):
        torch.manual_seed(19)
        images = torch.randn(2, 4, 3, 16, 16)
        masks = torch.zeros(2, regions, 3, 16, 16)
        masks[:, 0, :, 2:10, 3:11] = 1
        return images, masks, masks.flatten(2).sum(-1) > 0

    def test_missing_regions_never_enter_encoder_or_change_batchnorm(self):
        model = self.model().train()
        reference = copy.deepcopy(model).train()
        images, masks, present = self.inputs()
        observed = []
        handle = model.encoder.register_forward_pre_hook(
            lambda module, args: observed.append(args[0].detach().clone()))
        try:
            result = model(images, masks, present)
        finally:
            handle.remove()
        # Removing entirely absent channels must leave the CNN batch unchanged.
        expected = reference(images, masks[:, :1], present[:, :1])
        self.assertEqual(len(observed), 1)
        self.assertEqual(observed[0].shape[0], 2)
        self.assertTrue((observed[0].flatten(1).abs().sum(-1) > 0).all())
        torch.testing.assert_close(result['scores'], expected['scores'], rtol=0, atol=0)
        for name, buffer in model.named_buffers():
            torch.testing.assert_close(buffer, dict(reference.named_buffers())[name], rtol=0, atol=0)
        self.assertEqual(int(result['local'][:, 1:].count_nonzero()), 0)

    def test_missing_masks_and_pixels_outside_all_regions_get_no_gradient(self):
        model = self.model().train()
        images, masks, present = self.inputs()
        images.requires_grad_()
        masks.requires_grad_()
        result = model(images, masks, present)
        result['scores'][:, 0].sum().backward()
        self.assertEqual(int(masks.grad[:, 1:].count_nonzero()), 0)
        union = masks.detach().amax(1).unsqueeze(1).expand_as(images) > 0
        self.assertEqual(int(images.grad[~union].count_nonzero()), 0)
        self.assertGreater(float(images.grad[union].abs().sum()), 0)

    def test_zero_valid_region_and_inconsistent_presence_are_rejected(self):
        model = self.model()
        images, masks, present = self.inputs()
        with self.assertRaisesRegex(ValueError, 'Presence'):
            model(images, torch.zeros_like(masks), torch.zeros_like(present))
        with self.assertRaisesRegex(ValueError, 'Presence'):
            model(images, masks, torch.ones_like(present))
        with self.assertRaisesRegex(ValueError, 'at least one valid region'):
            masked_mean(torch.ones(2, 5, 8), torch.zeros(2, 5, dtype=torch.bool))

    def test_one_valid_graph_node_is_finite_with_no_edges(self):
        model = self.model(graph=True).train()
        images, masks, present = self.inputs()
        result = model(images, masks, present)
        for key in ('scores', 'local', 'adjacency', 'graph_loss'):
            self.assertTrue(torch.isfinite(result[key]).all(), key)
        self.assertEqual(int(result['adjacency'].count_nonzero()), 0)
        self.assertEqual(float(result['graph_loss'].detach()), 0)
        (result['scores'].sum() + result['graph_loss']).backward()
        for name, parameter in model.named_parameters():
            if parameter.grad is not None:
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_graph_missing_rows_columns_and_self_edges_are_zero(self):
        model = self.model(graph=True).train()
        images, masks, present = self.inputs()
        masks[0, 2, :, 10:14, 10:14] = 1
        present = masks.flatten(2).sum(-1) > 0
        result = model(images, masks, present)
        adjacency = result['adjacency']
        allowed = present[:, :, None] & present[:, None, :]
        self.assertEqual(int(adjacency[~allowed].count_nonzero()), 0)
        self.assertEqual(int(adjacency.diagonal(dim1=1, dim2=2).count_nonzero()), 0)
        self.assertEqual(float(adjacency[0, 0, 2].detach()), 1)
        self.assertEqual(float(adjacency[0, 2, 0].detach()), 1)
        self.assertEqual(int(adjacency[1].count_nonzero()), 0)
        self.assertTrue(torch.isfinite(result['graph_loss']))

    def test_four_arms_have_the_same_parameter_budget_and_shapes(self):
        counts = []
        shapes = []
        for method in METHODS:
            model = self.model(method)
            counts.append(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))
            shapes.append({name: tuple(parameter.shape) for name, parameter in model.named_parameters()})
        self.assertEqual(len(METHODS), 4)
        self.assertGreater(counts[0], 0)
        self.assertEqual(len(set(counts)), 1)
        self.assertTrue(all(shape == shapes[0] for shape in shapes))

    def test_fieldwise_loss_still_supervises_a_difficult_positive_field(self):
        vocab = [{'field': 'a'}, {'field': 'a'}, {'field': 'b'}, {'field': 'b'}]
        positives = torch.tensor([[True, False, True, False]])
        known = torch.ones_like(positives)
        mass_scores = torch.tensor([[12., -12., -4., 4.]], dtype=torch.float64, requires_grad=True)
        field_scores = mass_scores.detach().clone().requires_grad_()
        mass = masked_nce(mass_scores, positives, known, symmetric=False)
        fields = fieldwise_loss(field_scores, positives, known, vocab)
        mass.backward()
        fields.backward()
        # The easy field can satisfy set mass while the second field is wrong.
        self.assertLess(float(mass.detach()), 0.001)
        self.assertGreater(float(fields.detach()), 3.9)
        self.assertLess(abs(float(mass_scores.grad[0, 2])), 1e-5)
        self.assertLess(float(field_scores.grad[0, 2]), -0.49)
        self.assertGreater(float(field_scores.grad[0, 3]), 0.49)

    def test_unknown_labels_and_fully_missing_patients_have_zero_gradient(self):
        vocab = [{'field': 'a'}, {'field': 'a'}, {'field': 'b'}, {'field': 'b'}]
        positives = torch.tensor([[True, False, False, False], [False, False, False, False]])
        known = torch.tensor([[True, True, False, False], [False, False, False, False]])
        initial = torch.tensor([[0.3, -0.1, 100., -100.], [1., 2., 3., 4.]], dtype=torch.float64)
        for method in METHODS:
            with self.subTest(method=method):
                model = self.model(method)
                scores = initial.clone().requires_grad_()
                loss = model.loss({'scores': scores}, positives, known, vocab)
                loss.backward()
                self.assertTrue(torch.isfinite(loss))
                self.assertEqual(int(scores.grad[~known].count_nonzero()), 0)
                self.assertGreater(float(scores.grad[known].abs().sum()), 0)
                changed = initial.clone()
                changed[~known] = -12345
                torch.testing.assert_close(model.loss({'scores': changed}, positives, known, vocab),
                                           loss.detach(), rtol=0, atol=0)
        empty_scores = initial.clone().requires_grad_()
        empty_loss = fieldwise_loss(empty_scores, torch.zeros_like(positives), torch.zeros_like(known), vocab)
        empty_loss.backward()
        self.assertEqual(float(empty_loss.detach()), 0)
        self.assertEqual(int(empty_scores.grad.count_nonzero()), 0)

    def test_validation_rng_replays_scores_and_restores_all_cpu_streams(self):
        model = self.model().eval()
        images, masks, present = self.inputs()
        torch.manual_seed(101)
        np.random.seed(102)
        random.seed(103)
        torch_state, numpy_state, python_state = torch.get_rng_state().clone(), np.random.get_state(), random.getstate()

        def score():
            with validation_rng(42, 'cpu'), torch.no_grad():
                values = model(images, masks, present)['scores']
                return values + torch.randn_like(values) + float(np.random.rand()) + random.random()

        first, second = score(), score()
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_state))
        after_numpy = np.random.get_state()
        self.assertEqual(after_numpy[0], numpy_state[0])
        np.testing.assert_array_equal(after_numpy[1], numpy_state[1])
        self.assertEqual(after_numpy[2:], numpy_state[2:])
        self.assertEqual(random.getstate(), python_state)

        with self.assertRaisesRegex(RuntimeError, 'synthetic failure'):
            with validation_rng(42, 'cpu'):
                torch.randn(3)
                np.random.rand()
                random.random()
                raise RuntimeError('synthetic failure')
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_state))
        np.testing.assert_array_equal(np.random.get_state()[1], numpy_state[1])
        self.assertEqual(random.getstate(), python_state)


if __name__ == '__main__':
    unittest.main()
