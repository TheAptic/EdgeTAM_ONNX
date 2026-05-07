import unittest

import numpy as np

from scripts.prompt_inputs import build_prompt_arrays


class PromptInputsTests(unittest.TestCase):
    def test_variable_length_uses_active_points_only(self):
        pts = [(10, 10, 1)]
        coords, labels = build_prompt_arrays(pts, max_points=100, fixed_length=False)
        self.assertEqual(coords.shape, (1, 1, 2))
        self.assertEqual(labels.shape, (1, 1))
        self.assertEqual(float(coords[0, 0, 0]), 10.0)
        self.assertEqual(float(labels[0, 0]), 1.0)

    def test_fixed_length_pads_with_not_a_point(self):
        pts = [(10, 10, 1)]
        coords, labels = build_prompt_arrays(pts, max_points=4, fixed_length=True)
        self.assertEqual(coords.shape, (1, 4, 2))
        self.assertTrue(np.all(labels[0, 1:] == -1))

    def test_non_positive_labels_are_encoded_as_negative(self):
        pts = [(10, 10, -1), (20, 20, 0)]
        _, labels = build_prompt_arrays(pts, max_points=4, fixed_length=True)
        self.assertEqual(float(labels[0, 0]), 0.0)
        self.assertEqual(float(labels[0, 1]), 0.0)

    def test_positive_enum_labels_are_encoded_as_positive(self):
        pts = [(10, 10, 2)]
        _, labels = build_prompt_arrays(pts, max_points=4, fixed_length=True)
        self.assertEqual(float(labels[0, 0]), 1.0)


if __name__ == "__main__":
    unittest.main()
