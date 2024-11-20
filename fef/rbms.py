from typing import Dict
import torch
from abc import ABC, abstractmethod

from annadca.binary import annaRBMbin
from annadca.categorical import annaRBMcat
from annadca.binary.sampling import _sample_hiddens as _sample_hiddens_bin
from annadca.binary.sampling import _sample_visibles as _sample_visibles_bin
from annadca.binary.sampling import _sample_labels as _sample_labels_bin
from annadca.categorical.sampling import _sample_hiddens as _sample_hiddens_cat
from annadca.categorical.sampling import _sample_visibles as _sample_visibles_cat
from annadca.categorical.sampling import _sample_labels as _sample_labels_cat

class fefRBM(ABC):
    """Abstract class for the F&F Restricted Boltzmann Machine"""
    
    def __init__(self, **kwargs):
        pass
    
    @abstractmethod
    def sample_given_visible(
        self,
        gibbs_steps: int,
        chains: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        pass
    
    @abstractmethod
    def sample_given_label(
        self,
        gibbs_steps: int,
        chains: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        pass
        

class fefRBMbin(fefRBM, annaRBMbin):
    def __init__(self, **kwargs):
        annaRBMbin.__init__(self, **kwargs)
    
    def sample_given_visible(
        self,
        gibbs_steps: int,
        chains: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Samples the hidden units and the labels clamping the visible units.

        Args:
            gibbs_steps (int): Number of Alternating Gibbs Sampling steps.
            chains (Dict[str, torch.Tensor]): Markov chains to be updated.

        Returns:
            Dict[str, torch.Tensor]: Updated chains.
        """
        for _ in range(gibbs_steps):
            chains["hidden"], _ = _sample_hiddens_bin(
                visible=chains["visible"],
                label=chains["label"],
                params=self.params,
            )
            chains["label"], _ = _sample_labels_bin(
                hidden=chains["hidden"],
                params=self.params,
            )
        
        return chains
    
    
    def sample_given_label(
        self,
        gibbs_steps: int,
        chains: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Samples the hidden units and the visible units clamping the label units.

        Args:
            gibbs_steps (int): Number of Alternating Gibbs Sampling steps.
            chains (Dict[str, torch.Tensor]): Markov chains to be updated.

        Returns:
            Dict[str, torch.Tensor]: Updated chains.
        """
        for _ in range(gibbs_steps):
            chains["hidden"], _ = _sample_hiddens_bin(
                visible=chains["visible"],
                label=chains["label"],
                params=self.params,
            )
            chains["visible"], _ = _sample_visibles_bin(
                hidden=chains["hidden"],
                params=self.params,
            )
        
        return chains
    

class fefRBMcat(fefRBM, annaRBMcat):
    def __init__(self, **kwargs):
        annaRBMcat.__init__(self, **kwargs)
    
    def sample_given_visible(
        self,
        gibbs_steps: int,
        chains: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Samples the hidden units and the labels clamping the visible units.

        Args:
            gibbs_steps (int): Number of Alternating Gibbs Sampling steps.
            chains (Dict[str, torch.Tensor]): Markov chains to be updated.

        Returns:
            Dict[str, torch.Tensor]: Updated chains.
        """
        for _ in range(gibbs_steps):
            chains["hidden"], _ = _sample_hiddens_cat(
                visible=chains["visible"],
                label=chains["label"],
                params=self.params,
            )
            chains["label"], _ = _sample_labels_cat(
                hidden=chains["hidden"],
                params=self.params,
            )
        
        return chains
    
    
    def sample_given_label(
        self,
        gibbs_steps: int,
        chains: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Samples the hidden units and the visible units clamping the label units.

        Args:
            gibbs_steps (int): Number of Alternating Gibbs Sampling steps.
            chains (Dict[str, torch.Tensor]): Markov chains to be updated.

        Returns:
            Dict[str, torch.Tensor]: Updated chains.
        """
        for _ in range(gibbs_steps):
            chains["hidden"], _ = _sample_hiddens_cat(
                visible=chains["visible"],
                label=chains["label"],
                params=self.params,
            )
            chains["visible"], _ = _sample_visibles_cat(
                hidden=chains["hidden"],
                params=self.params,
            )
        
        return chains
