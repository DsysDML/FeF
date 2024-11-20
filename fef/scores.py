from typing import Tuple, Dict, Callable
import torch
from torch.nn import MSELoss
import numpy as np
import gzip
from adabmDCA.stats import get_freq_single_point as get_freq_single_point_cat
from adabmDCA.stats import get_freq_two_points as get_freq_two_points_cat
from annadca.binary.stats import get_freq_single_point as get_freq_single_point_bin
from annadca.binary.stats import get_freq_two_points as get_freq_two_points_bin


from fef.rbms import fefRBM

def LL_score(
    rbm: fefRBM,
    data: torch.Tensor,
    gen: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    """Computes the difference between the Log-Likelihoods of the data and the generated samples
    according to the rbm model.

    Args:
        rbm (fefRBM): RBM model.
        data (torch.Tensor): Data matrix.
        gen (torch.Tensor): Generated samples.

    Returns:
        float: MSE of the Log-Likelihoods.
    """
    
    ll_data = rbm.compute_energy_visibles(visible=data, label=labels).mean().item()
    ll_gen = rbm.compute_energy_visibles(visible=gen, label=labels).mean().item()
    
    return (ll_gen / ll_data - 1.)**2


@torch.jit.script
def _AAI_score(
    data: torch.Tensor,
    gen: torch.Tensor,
) -> Tuple[float, float]:
    conc = torch.cat([data, gen], 0)
    # all-to-all distance matrix
    dAB = torch.cdist(conc, conc, p=0)
    torch.diagonal(dAB).fill_(float('inf'))

    # the next line is used to tranform the matrix
    #  d_TT d_TF   INTO d_TF- d_TT-  where the minus indicate a reverse order of the columns
    #  d_FT d_FF        d_FT  d_FF
    dAB[:dAB.shape[0] // 2, :] = torch.flip(dAB[:int(dAB.shape[0] / 2),:], dims=[1])
    closest = dAB.argmin(axis=1)
    n = int(closest.shape[0] / 2)

    # correctly_classified = closest >= n
    AAI_data = (closest[:n] >= n).sum() / n  # for a true sample, prob that the closest is in the set of true samples
    AAI_gen = (closest[n:] >= n).sum() / n  # for a fake sample, prob that the closest is in the set of fake samples

    return (AAI_data.item(), AAI_gen.item())

def AAI_score(
    data: torch.Tensor,
    gen: torch.Tensor,
) -> Tuple[float, float]:
    """Computes the Adversarial Accuracy Indicator (AAI) scores of the data and the generated samples.

    Args:
        data (torch.Tensor): Data matrix.
        gen (torch.Tensor): Generated samples.
        
    Returns:
        Tuple[float, float]: AAI scores:
            AAI_data: for a true sample, prob that the closest is in the set of true samples.
            AAI_gen: for a fake sample, prob that the closest is in the set of fake samples.
    """
    return _AAI_score(data, gen)


def spectrum_score(
    spectrum_data: torch.Tensor,
    gen: torch.Tensor,
) -> float:
    """Computes the MSE between the spectra of the data and the generated samples matrices.

    Args:
        spectrum_data (torch.Tensor): Eigenvalues of the data matrix.
        gen (torch.Tensor): Generated samples.

    Returns:
        float: MSE between the two spectra.
    """
    num_samples = gen.shape[0]
    gen = gen.view(num_samples, -1)
    spectrum_gen = torch.linalg.svdvals(gen - gen.mean(0)).square() / num_samples
    
    return MSELoss()(spectrum_data, spectrum_gen).item()


def entropy_score(
    entropy_data: float,
    gen: torch.Tensor,
) -> float:
    """
    Computes the entropy score of the generated samples.
    
    Args:
        entropy_data (float): Per-sample entropy of the data.
        gen (torch.Tensor): Generated samples.
        
    Returns:
        float: Entropy score.
    """
    entropy_gen = len(gzip.compress(gen.int().cpu().numpy())) / len(gen)
    score = (entropy_gen / entropy_data) - 1.
    
    return score
    
    
def first_moment_score(
    fi_data: torch.Tensor,
    gen: torch.Tensor,
) -> float:
    """Computes the MSE between the single-point frequences of the data and the generated samples.

    Args:
        fi_data (torch.Tensor): Single-point frequences of the data.
        gen (torch.Tensor): Generated samples.

    Returns:
        float: MSE of the single-point frequences.
    """
    if fi_data.dim() == 1:
        fi_gen = get_freq_single_point_bin(gen, weights=None, pseudocount=1e-8)
    elif fi_data.dim() == 2:
        fi_gen = get_freq_single_point_cat(gen, weights=None, pseudocount=1e-8)
    else:
        raise ValueError("The input data must be either binary or one-hot for categorical.")
    
    return MSELoss()(fi_data, fi_gen).item()


def second_moment_score(
    cov_data: torch.Tensor,
    gen: torch.Tensor,
) -> float:
    """Computes the MSE between the two-point frequences of the data and the generated samples.

    Args:
        cov_data (torch.Tensor): Covariance matrix of the data.
        gen (torch.Tensor): Generated samples.

    Returns:
        float: MSE of the two-point frequences.
    """
    if cov_data.dim() == 2:
        fi_gen = get_freq_single_point_bin(gen, weights=None, pseudocount=1e-8)
        fij_gen = get_freq_two_points_bin(gen, weights=None, pseudocount=1e-8)
        cov_gen = fij_gen - torch.einsum('i,j->ij', fi_gen, fi_gen)
    elif cov_data.dim() == 4:
        fi_gen = get_freq_single_point_cat(gen, weights=None, pseudocount=1e-8)
        fij_gen = get_freq_two_points_cat(gen, weights=None, pseudocount=1e-8)
        cov_gen = fij_gen - torch.einsum('ij,kl->ijkl', fi_gen, fi_gen)
    else:
        raise ValueError("The input data must be either binary or one-hot for categorical.")
    
    return MSELoss()(cov_data, cov_gen).item()


class Score():
    def __init__(
        self,
        checkpoints: np.ndarray | list | None = None,
        record_times: np.ndarray | list | None = None,
        labels: np.ndarray | list | None = None,
        score_function: Callable | None = None,
) -> None:
        
        self.checkpoints = checkpoints
        self.record_times = record_times
        self.labels = labels
        self.score_function = score_function
        
    def load(self, filename : str) -> None:
        pass
    
    def save(self, filename : str) -> None:
        pass

    def update(self, checkpoint : int, score : float) -> None:
        pass
            
            
            
        