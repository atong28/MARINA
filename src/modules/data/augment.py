import numpy as np
import torch


class PeakAugmenter:
    """
    Applies peak injection and/or peak dropout to NMR tensors during training.

    Augmented modalities: hsqc, c_nmr, h_nmr. mass_spec is intentionally excluded.

    Injection: with prob p_add, draw k ~ Poisson(alpha_add * N) and append k rows
               sampled from the training-set empirical distribution.
    Dropout:   with prob p_remove, draw k ~ Poisson(alpha_remove * N) and drop k
               randomly chosen rows (always keeping at least 1).
    """

    AUGMENTED_MODALITIES = frozenset({'hsqc', 'c_nmr', 'h_nmr'})

    def __init__(
        self,
        dist_path: str,
        p_add: float,
        alpha_add: float,
        p_remove: float,
        alpha_remove: float,
    ):
        data = np.load(dist_path)
        self.dist = {k: data[k] for k in data.files}
        self.p_add = p_add
        self.alpha_add = alpha_add
        self.p_remove = p_remove
        self.alpha_remove = alpha_remove

    def augment(self, tensor: torch.Tensor, modality: str) -> torch.Tensor:
        if modality not in self.AUGMENTED_MODALITIES:
            return tensor
        if self.p_remove > 0 and torch.rand(1).item() < self.p_remove:
            tensor = self._remove(tensor)
        if self.p_add > 0 and torch.rand(1).item() < self.p_add:
            tensor = self._add(tensor, modality)
        return tensor

    def _add(self, tensor: torch.Tensor, modality: str) -> torch.Tensor:
        N = tensor.shape[0]
        k = int(np.random.poisson(self.alpha_add * N))
        if k == 0:
            return tensor
        dist = self.dist[modality]
        indices = np.random.randint(0, len(dist), size=k)
        synthetic = torch.from_numpy(dist[indices]).to(dtype=tensor.dtype)
        return torch.cat([tensor, synthetic], dim=0)

    def _remove(self, tensor: torch.Tensor) -> torch.Tensor:
        N = tensor.shape[0]
        k = min(int(np.random.poisson(self.alpha_remove * N)), N - 1)
        if k <= 0:
            return tensor
        keep = torch.randperm(N)[: N - k]
        return tensor[keep]
