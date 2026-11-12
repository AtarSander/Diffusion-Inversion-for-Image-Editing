import importlib.util
from pathlib import Path
import unittest

import numpy as np

spec = importlib.util.spec_from_file_location(
    'prepare_bbbc021', Path(__file__).parents[1] / 'scripts' / 'prepare_bbbc021.py')
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


class PreparationTests(unittest.TestCase):
    def test_channel_scaling(self):
        a = np.array([[0, 25], [50, 100]], dtype=np.uint16)
        expected = np.array([[0, 64], [128, 255]], dtype=np.uint8)
        np.testing.assert_array_equal(prepare.channel_uint8(a, [0, 100]), expected)

    def test_invalid_channels_rejected(self):
        for a in (np.zeros((4, 4)), np.array([[np.nan, 0], [1, 2]]), np.zeros((2, 2, 3))):
            with self.assertRaises(ValueError):
                prepare.channel_uint8(a, [0.1, 99.9])


if __name__ == '__main__':
    unittest.main()
