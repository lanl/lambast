import torch
import numpy as np
from typing import Union

TensorLike = Union[np.ndarray, torch.Tensor]

def _to_float_tensor(x: TensorLike) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        t = x
    else:
        t = torch.from_numpy(np.asarray(x))
    return t.float().contiguous()