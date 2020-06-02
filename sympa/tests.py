
import torch
import unittest
from sympa.utils import set_seed


class TestCase(unittest.TestCase):

    def setUp(self):
        set_seed(42)

    def assertAllClose(self, a: torch.tensor, b: torch.tensor, rtol=1e-05, atol=1e-08):
        return torch.allclose(a, b, rtol=rtol, atol=atol)

    def assertAllEqual(self, a: torch.tensor, b: torch.tensor):
        return torch.eq(a, b).all()
