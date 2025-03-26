import numpy as np
import torch


def torch_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def numpy_to_torch(array: np.ndarray, device: str = "cpu") -> torch.Tensor:
    return torch.as_tensor(array, device=device)
