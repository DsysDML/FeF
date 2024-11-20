import torch
from torch.nn import MSELoss
import numpy as np
import gzip

def LL_score_Ssl(D : torch.Tensor, G : torch.Tensor, rbm) -> torch.Tensor:
    """Computes the difference between the unnormalized Log-Likelihoods of the data D and the generated samples G according to the rbm model.

    Args:
        D (torch.Tensor): Data matrix of shape (# samples, Nv).
        LD (torch.Tensor): Labels associated with the data.
        G (torch.Tensor): Generated samples matrix of shape (# samples, Nv).
        LG (torch.Tensor): Labels associated with the generated samples.
        rbm (RBM) : RBM model.

    Returns:
        torch.Tensor: MSE of the Log-Likelihoods.
    """
    
    LLD = torch.mean(rbm.compute_energy_visibles(D))
    LLG = torch.mean(rbm.compute_energy_visibles(G))
    
    return (LLG / LLD - 1.)**2

def AAI_score(D : torch.Tensor, G : torch.Tensor) -> torch.Tensor:
    """Computes the Adversarial Accuracy Indicator (AAI) scores of the data D and the generated samples G.
    In case of categorical variables, the input matrices must be the one-hot representations of the original ones.

    Args:
        D (torch.Tensor): Data matrix of shape (# samples, Nv) if binary, and (# samples, Nv * num_states) if categorical.
        G (torch.Tensor): Generated samples matrix of shape (# samples, Nv) if binary, and (# samples, Nv * num_states) if categorical.

    Returns:
        list: (AAI_data, AAI_generated) scores.
    """
    
    conc = torch.cat([torch.round(D), torch.round(G)], 0) # If data are continuous, turn them into binary variables
    dAB = torch.cdist(conc, conc, p=0)
    torch.diagonal(dAB).fill_(float('inf'))

    # the next line is use to tranform the matrix
    #  d_TT d_TF   INTO d_TF- d_TT-  where the minus indicate a reverse order of the columns
    #  d_FT d_FF        d_FT  d_FF
    dAB[:dAB.shape[0] // 2, :] = torch.flip(dAB[:int(dAB.shape[0] / 2),:], dims=[1])
    closest = dAB.argmin(axis=1)
    n = int(closest.shape[0] / 2)

    # correctly_classified = closest >= n
    AAdata = (closest[:n] >= n).sum() / n  # for a true sample, prob that the closest is in the set of true samples
    AAgen = (closest[n:] >= n).sum() / n  # for a fake sample, prob that the closest is in the set of fake samples

    return AAdata, AAgen

def spectrum_score(D : torch.Tensor, G : torch.Tensor) -> torch.Tensor:
    """Computes the MSE between the spectra of the data D and the generated samples G.
    In case of categorical variables, the input matrices must be the one-hot representations of the original ones.

    Args:
        D (torch.Tensor): Data matrix of shape (# samples, Nv) if binary, and (# samples, Nv * num_states) if categorical.
        G (torch.Tensor): Generated samples matrix of shape (# samples, Nv) if binary, and (# samples, Nv * num_states) if categorical.

    Returns:
        torch.Tensor: MSE of the spectra.
    """
    N = torch.tensor(G.shape[0])
    spectrum_D = torch.linalg.svdvals(D - D.type(torch.float32).mean(0)) * torch.pow(N, -0.5)
    spectrum_G = torch.linalg.svdvals(G - G.type(torch.float32).mean(0)) * torch.pow(N, -0.5)
    
    return MSELoss()(spectrum_D, spectrum_G)

def entropy_score(entropy_D : float, G : torch.Tensor):
    """
    Computes the entropy score.
    
    Args:
        entropy_D (float): Entropy-per-sample of the data.
        G (torch.Tensor): Generated samples.
    """
    entropy_G = len(gzip.compress(G.int().cpu().numpy())) / len(G)

    score = (entropy_G / entropy_D) - 1.
    
    return score
    
def first_moment_score(D : torch.Tensor, G : torch.Tensor) -> torch.Tensor:
    """Computes the MSE between the frequences of the data D and the generated samples G.
    In case of categorical variables, the input matrices must be the one-hot representations of the original ones.

    Args:
        D (torch.Tensor): Data matrix of shape (# samples, Nv) if binary, and (# samples, Nv * num_states) if categorical.
        G (torch.Tensor): Generated samples matrix of shape (# samples, Nv) if binary, and (# samples, Nv * num_states) if categorical.

    Returns:
        torch.Tensor: MSE of the frequences.
    """
    
    fD = D.type(torch.float32).mean(0)
    fG = G.type(torch.float32).mean(0)
    
    return MSELoss()(fD, fG)

def second_moment_score(D : torch.Tensor, G : torch.Tensor) -> torch.Tensor:
    """Computes the MSE between the covariance matrix of the data D and the generated samples G.
    The covariance matrices are computed one row at a time in order to fit the memory contraints.
    In case of categorical variables, the input matrices must be the one-hot representations of the original ones.

    Args:
        D (torch.Tensor): Data matrix of shape (# samples, Nv) if binary, and (# samples, Nv * num_states) if categorical.
        G (torch.Tensor): Generated samples matrix of shape (# samples, Nv) if binary, and (# samples, Nv * num_states) if categorical.

    Returns:
        torch.Tensor: MSE of the two covariance matrcies.
    """
    
    # center the datasets and take the transposed
    DcT = (D - D.type(torch.float32).mean(0)).mT
    GcT = (G - G.type(torch.float32).mean(0)).mT
    N, M = D.shape
    
    l2_loss = 0.
    for i in range(M):
        CiD = DcT @ DcT[i].t() / (N - 1)
        CiG = GcT @ GcT[i].t() / (N - 1)
        l2_loss += MSELoss()(CiD, CiG) / M
    
    return l2_loss

##############################################################################
# Batched versions of the scores above

def LL_score_batched(D : torch.Tensor, G : torch.Tensor, rbm, batch_size : int) -> torch.Tensor:
    """Computes the difference between the unnormalized Log-Likelihoods of the data D and the generated samples G according to the rbm model.

    Args:
        D (torch.Tensor): Data matrix of shape (# samples, Nv).
        G (torch.Tensor): Generated samples matrix of shape (# samples, Nv).
        rbm (RBM) : RBM model.
        batch_size (int): Batch size.

    Returns:
        torch.Tensor: MSE of the Log-Likelihoods.
    """
    LLD = []
    LLG = []
    
    n_epochs = D.shape[0] // batch_size + 1
    for ep in range(n_epochs):
        D_ep = D[ep * batch_size : (ep + 1) * batch_size]
        G_ep = G[ep * batch_size : (ep + 1) * batch_size]
        LLD.append(torch.mean(rbm.compute_energy_visibles(D_ep)).unsqueeze(0))
        LLG.append(torch.mean(rbm.compute_energy_visibles(G_ep)).unsqueeze(0))
    LLG = torch.cat(LLG, 0).mean()
    LLD = torch.cat(LLD, 0).mean()
    
    return (LLG / LLD - 1.)**2

def LL_score_batched_Ssl(D : torch.Tensor, LD : torch.Tensor, G : torch.Tensor, LG : torch.Tensor, rbm, batch_size : int) -> torch.Tensor:
    """Computes the difference between the unnormalized Log-Likelihoods of the data D and the generated samples G according to the rbm model.

    Args:
        D (torch.Tensor): Data matrix of shape (# samples, Nv).
        LD (torch.Tensor): Labels associated with the data.
        G (torch.Tensor): Generated samples matrix of shape (# samples, Nv).
        LG (torch.Tensor): Labels associated with the generated samples.
        rbm (RBM) : RBM model.
        batch_size (int): Batch size.

    Returns:
        torch.Tensor: MSE of the Log-Likelihoods.
    """
    LLD = []
    LLG = []
    
    n_epochs = D.shape[0] // batch_size + 1
    for ep in range(n_epochs):
        D_ep = D[ep * batch_size : (ep + 1) * batch_size]
        LD_ep = LD[ep * batch_size : (ep + 1) * batch_size]
        G_ep = G[ep * batch_size : (ep + 1) * batch_size]
        LG_ep = LG[ep * batch_size : (ep + 1) * batch_size]
        LLD.append(torch.mean(rbm.compute_energy_visibles(D_ep, LD_ep)).unsqueeze(0))
        LLG.append(torch.mean(rbm.compute_energy_visibles(G_ep, LG_ep)).unsqueeze(0))
    LLG = torch.cat(LLG, 0).mean()
    LLD = torch.cat(LLD, 0).mean()
    
    return (LLG / LLD - 1.)**2

class SCORE():
    
    def __init__(self, t_ages : np.ndarray, record_times : np.ndarray, labels : np.ndarray) -> None:
        
        self.t_ages = t_ages
        self.record_times = record_times
        self.labels = labels
        self.n_labels = len(labels)
        self.mean_temp = 0
        self.counter = 0
        self.score_timeline = {t : {l : [] for l in self.labels} for t in self.t_ages}
        self.mean = {t : [] for t in self.t_ages}

    def update(self, t_age : int, label : str, value : float) -> None:
        
        self.mean_temp += value
        self.score_timeline[t_age][label].append(value)
        self.counter += 1
        if self.counter == self.n_labels:
            self.mean[t_age].append(self.mean_temp / self.n_labels)
            self.counter = 0
            self.mean_temp = 0
            
    def get_label_score(self, t_age : int, label : str) -> list:
        
        return self.score_timeline[t_age][label]
    
    def get_mean_score(self, t_age : int) -> list:
        
        return self.mean[t_age]
    

class SCORE_SINGLE():
    
    def __init__(self, t_ages : np.ndarray, record_times : np.ndarray) -> None:
        
        self.t_ages = t_ages
        self.record_times = record_times
        self.mean = {t : [] for t in self.t_ages}

    def update(self, t_age : int, value : float) -> None:
        
        self.mean[t_age].append(value)
    
    def get_mean_score(self, t_age : int) -> list:
        
        return self.mean[t_age]
            
            
            
        