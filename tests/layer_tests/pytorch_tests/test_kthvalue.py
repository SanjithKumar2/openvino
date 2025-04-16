import pytest
from pytorch_layer_test_class import PytorchLayerTest
import torch
import numpy as np

class TestKthValue(PytorchLayerTest):
    def _prepare_input(self, shape=(4, 5), k=2, dim=1, keepdim=False, dtype=np.float32):
        x = np.random.uniform(0, 100, size=shape).astype(dtype)
        k_tensor = np.array(k, dtype=np.int64)
        dim_tensor = np.array(dim, dtype=np.int64)
        keepdim_tensor = np.array(keepdim, dtype=np.bool_)
        return (x, k_tensor, dim_tensor, keepdim_tensor)

    def create_model(self, keepdim=False):
        class aten_kthvalue(torch.nn.Module):
            def forward(self, x, k, dim, keepdim):
                return torch.kthvalue(x, k.item(), dim.item(), keepdim.item())

        return aten_kthvalue(), None, "aten::kthvalue"

    @pytest.mark.parametrize("shape", [(4, 5), (3, 6, 2)])
    @pytest.mark.parametrize("dim", [0, 1])
    @pytest.mark.parametrize("k", [1, 2])
    @pytest.mark.parametrize("keepdim", [True, False])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_kthvalue(self, shape, dim, k, keepdim, ie_device, precision, ir_version):
        self._test(*self.create_model(keepdim), ie_device, precision, ir_version,
                   kwargs_to_prepare_input={
                       "shape": shape,
                       "k": k,
                       "dim": dim,
                       "keepdim": keepdim
                   })