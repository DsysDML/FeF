from typing import Dict
import torch

from fef.rbms import fefRBM


def fef(
    rbm: fefRBM,
    data_batch: Dict[str, torch.Tensor],
    gibbs_steps: int,
    optimizer: torch.optim.Optimizer,
    pseudo_count: float = 0.0,
    centered: bool = True,
    eta: float = 1.0,
    single_gradient: bool = False,
) -> Dict[str, torch.Tensor]:
    """Computes the gradient of the parameters of the model and the Markov chains using the F&F algorithm.

    Args:
        rbm (fefRBM): RBM model to be trained.
        data_batch (Dict[str, torch.Tensor]): Batch of data.
        gibbs_steps (int): Number of Alternating Gibbs Sampling steps.
        pseudo_count (float, optional): Pseudo count to be added to the data frequencies. Defaults to 0.0.
        centered (bool, optional): Whether to use centered gradients. Defaults to True.
        eta (float, optional): Relative contribution of the label term. Defaults to 1.0.

    Returns:
        Dict[str, torch.Tensor]: Updated chains.
    """
    num_samples = data_batch["visible"].shape[0]
    
    # Compute the hidden magnetization of the data
    data_batch["hidden"] = rbm.sample_hiddens(**data_batch)["hidden_mag"]
    
    # For all data in the batch that have zero label, infer the label
    infered_labels = rbm.sample_labels(hidden=data_batch["hidden"])["label_mag"]
    data_batch["label"] = torch.where(torch.all(data_batch["label"] == 0, dim=1, keepdim=True), infered_labels, data_batch["label"])
    
    # Initialize the chains at random
    chains_clamp_label = rbm.init_chains(num_samples=num_samples)
    chains_clamp_label["label"] = data_batch["label"]
    chains_clamp_visible = rbm.init_chains(num_samples=num_samples)
    chains_clamp_visible["visible"] = data_batch["visible"]
    
    # Update the chains
    chains_clamp_label = rbm.sample_given_label(gibbs_steps=gibbs_steps, chains=chains_clamp_label)
    chains_clamp_visible = rbm.sample_given_visible(gibbs_steps=gibbs_steps, chains=chains_clamp_visible)
    
    # Compute the ooe gradient with clamped labels
    rbm.compute_gradient(
        data=data_batch,
        chains=chains_clamp_label,
        pseudo_count=pseudo_count,
        centered=centered,
        eta=eta,
    )
    optimizer.step()
    optimizer.zero_grad(set_to_none=False)
    
    if not single_gradient:
        # Compute the ooe gradient with clamped visible units
        rbm.compute_gradient(
            data=data_batch,
            chains=chains_clamp_visible,
            pseudo_count=pseudo_count,
            centered=centered,
            eta=eta,
        )
        optimizer.step()
        optimizer.zero_grad(set_to_none=False)