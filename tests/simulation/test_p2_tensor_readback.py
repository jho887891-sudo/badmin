"""No Isaac/GPU required: simulate a CUDA tensor that rejects np.asarray()."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "simulation"))
from p2_tensor_readback import host_floats


class FakeCudaTensor:
    def __init__(self, values, detached=False, on_cpu=False):
        self.values = values
        self.detached = detached
        self.on_cpu = on_cpu

    def __array__(self, *args, **kwargs):
        raise TypeError("can't convert cuda:0 device type tensor to numpy")

    def detach(self):
        return FakeCudaTensor(self.values, detached=True, on_cpu=self.on_cpu)

    def cpu(self):
        if not self.detached:
            raise AssertionError("detach() must precede cpu()")
        return FakeCudaTensor(self.values, detached=True, on_cpu=True)

    def reshape(self, *args):
        if not self.on_cpu:
            raise AssertionError("reshape must be done after .cpu()")
        return self

    def tolist(self):
        if not self.on_cpu:
            raise AssertionError("tolist must be done after .cpu()")
        return self.values


class CudaReadbackContractTests(unittest.TestCase):
    def test_device_tensor_is_moved_to_cpu(self):
        self.assertEqual(host_floats(FakeCudaTensor([.12, -.05, .2]), 3),
                         [.12, -.05, .2])

    def test_truncates_to_requested_count(self):
        self.assertEqual(host_floats(FakeCudaTensor([1, 0, 0, 0, 999]), 4),
                         [1., 0., 0., 0.])

    def test_python_vector_without_torch(self):
        self.assertEqual(host_floats((1, 2, 3), 3), [1., 2., 3.])

    def test_incomplete_tensor_fails_closed(self):
        with self.assertRaises(ValueError):
            host_floats(FakeCudaTensor([1, 2]), 3)

    def test_nan_fails_closed(self):
        with self.assertRaises(ValueError):
            host_floats(FakeCudaTensor([1, float("nan"), 3]), 3)

    def test_invalid_count_fails(self):
        with self.assertRaises(ValueError):
            host_floats([0], 0)


if __name__ == "__main__":
    unittest.main()
