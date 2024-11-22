from typing import Tuple, Dict, Callable
import torch
from torch.nn import MSELoss
import numpy as np
import h5py
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
    dAB = torch.cdist(conc, conc, p=0.0)
    torch.diagonal(dAB).fill_(float('inf'))

    # the next line is used to tranform the matrix
    #  d_TT d_TF   INTO d_TF- d_TT-  where the minus indicate a reverse order of the columns
    #  d_FT d_FF        d_FT  d_FF
    dAB[:dAB.shape[0] // 2, :] = torch.flip(dAB[:int(dAB.shape[0] / 2),:], dims=[1])
    closest = torch.argmin(dAB, dim=1)
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
    score = ((entropy_gen / entropy_data) - 1.)**2
    
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
        fi_gen = get_freq_single_point_bin(gen, weights=None, pseudo_count=1e-8)
    elif fi_data.dim() == 2:
        fi_gen = get_freq_single_point_cat(gen, weights=None, pseudo_count=1e-8)
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
        fi_gen = get_freq_single_point_bin(gen, weights=None, pseudo_count=1e-8)
        fij_gen = get_freq_two_points_bin(gen, weights=None, pseudo_count=1e-8)
        cov_gen = fij_gen - torch.einsum('i,j->ij', fi_gen, fi_gen)
    elif cov_data.dim() == 4:
        fi_gen = get_freq_single_point_cat(gen, weights=None, pseudo_count=1e-8)
        fij_gen = get_freq_two_points_cat(gen, weights=None, pseudo_count=1e-8)
        cov_gen = fij_gen - torch.einsum('ij,kl->ijkl', fi_gen, fi_gen)
    else:
        raise ValueError("The input data must be either binary or one-hot for categorical.")
    
    return MSELoss()(cov_data, cov_gen).item()


class Score():
    def __init__(
        self,
        filename: str | None = None,
        checkpoints: np.ndarray | list | None = None,
        record_times: np.ndarray | list | None = None,
        labels: np.ndarray | list | None = None,
        score_function: Callable | None = None,
) -> None:
        """Container for the scores used to evaluate the model.

        Args:
            checkpoints (np.ndarray | list | None, optional): List of checkpoints to analize that have
                been saved during the training. Defaults to None.
            record_times (np.ndarray | list | None, optional): List of sampling times at which the scores
                must be computed. Defaults to None.
            labels (np.ndarray | list | None, optional): List of unique labels present in the dataset. Defaults to None.
            score_function (Callable | None, optional): The scoring function to be used. Defaults to None.
        """
        
        self.checkpoints = checkpoints
        self.record_times = record_times
        self.labels = labels
        self.score_function = score_function
        self.records = {}
        if (self.labels is not None) and (self.checkpoints is not None) and (self.record_times is not None):
            self.checkpoint_to_index = {checkpoint: i for i, checkpoint in enumerate(self.checkpoints)}
            self.record_time_to_index = {record_time: i for i, record_time in enumerate(self.record_times)}
            for label in self.labels:
                self.records[label] = np.zeros(shape=(len(self.checkpoints), len(self.record_times)))
        
        if filename is not None:
            self.load(filename)
    
    
    def __getitem__(self, label: str) -> np.ndarray:
        return self.records[label]
    
    
    def __setitem__(self, key, value):
        self.records[key] = value
    
    
    def load(self, filename: str) -> None:
        with h5py.File(filename, 'r') as f:
            self.checkpoints = f["checkpoints"][()]
            self.record_times = f["record_times"][()]
            self.labels = [l for l in list(f.keys()) if l not in ["checkpoints", "record_times"]]
            self.checkpoint_to_index = {checkpoint: i for i, checkpoint in enumerate(self.checkpoints)}
            self.record_time_to_index = {record_time: i for i, record_time in enumerate(self.record_times)}
            for label in self.labels:
                self.records[label] = f[label][()]
    
    
    def save(self, filename: str) -> None:
        with h5py.File(filename, 'w') as f:
            f.create_dataset("checkpoints", data=self.checkpoints)
            f.create_dataset("record_times", data=self.record_times)
            for label in self.labels:
                f.create_dataset(label, data=self.records[label])
                
                
    def set_score_function(self, score_function: Callable) -> None:
        self.score_function = score_function
        self._update_evaluate_docstring()
        
    
    def _update_evaluate_docstring(self) -> None:
        if self.score_function is not None:
            self.evaluate.__doc__ = f"""Evaluates the model for a label at a given checkpoint and record time.

            Args:
                label (str): Label of the dataset.
                checkpoint (int): Checkpoint.
                record_time (int): Record time.
                kwargs: Additional arguments specific for the scoring function.

            Scoring function description:
            {self.score_function.__doc__}
            """

    
    def evaluate(
        self,
        label: str,
        checkpoint: int,
        record_time: int,
        **kwargs,
    ) -> None:
        """Evaluates the model for a label at a given checkpoint and record time.

        Args:
            label (str): Label of the dataset.
            checkpoint (int): Checkpoint.
            record_time (int): Record time.
            kwargs: Additional arguments specific for the scoring function.
        """
        if self.score_function is None:
            raise ValueError("The scoring function has not been set.")
        
        self.records[label][self.checkpoint_to_index[checkpoint], self.record_time_to_index[record_time]] = self.score_function(**kwargs)
            
    
    def mean_across_labels(self) -> np.ndarray:
        """Computes the mean score across all the labels.

        Returns:
            np.ndarray: Mean score.
        """
        return np.mean([self.records[label] for label in self.labels], axis=0)
    
    
    def std_across_labels(self) -> np.ndarray:
        """Computes the standard deviation of the score across all the labels.

        Returns:
            np.ndarray: Standard deviation.
        """
        return np.std([self.records[label] for label in self.labels], axis=0)
            
        