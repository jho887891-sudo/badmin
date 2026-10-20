"""Adapter to execute the 10 original pytest-style P0 tests with stdlib unittest.

Each function accepts only the pytest tmp_path fixture; we supply a fresh
TemporaryDirectory/Path to avoid adding a network dependency to P0 offline CI.
"""
from pathlib import Path
import tempfile
import unittest

import test_p0_subprocess_gate_runner as original


class P0ExternalRunnerContractTests(unittest.TestCase):
    pass


for _name in sorted(name for name in dir(original) if name.startswith("test_")):
    _test_func = getattr(original, _name)
    def _wrap(fn):
        def run_test(self):
            with tempfile.TemporaryDirectory(prefix="p0-unit-") as directory:
                fn(Path(directory))
        return run_test
    setattr(P0ExternalRunnerContractTests, _name, _wrap(_test_func))


if __name__ == "__main__":
    unittest.main()
