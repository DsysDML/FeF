import argparse
from pathlib import Path
import torch
import fef.scores as scores
from fef.parser import add_args_score
import gzip
import numpy as np
from fef.rbms import fefRBMbin, fefRBMcat

from annadca.utils import get_saved_updates
from annadca.dataset import DatasetBin, DatasetCat, get_dataset
from adabmDCA.utils import get_device
from adabmDCA.fasta_utils import get_tokens
from adabmDCA.stats import get_freq_single_point as get_freq_single_point_cat
from adabmDCA.stats import get_freq_two_points as get_freq_two_points_cat
from annadca.binary.stats import get_freq_single_point as get_freq_single_point_bin
from annadca.binary.stats import get_freq_two_points as get_freq_two_points_bin


# import command-line input arguments
def create_parser():
    # Important arguments
    parser = argparse.ArgumentParser(description="Scoring a fefRBM or annaRBM model")
    parser = add_args_score(parser) 
    
    return parser


if __name__ == '__main__':
    
    # Load parser, training dataset and DCA model
    parser = create_parser()
    args = parser.parse_args()
    
    if args.dtype == "float32":
        dtype = torch.float32
    elif args.dtype == "float64":
        dtype = torch.float64
    
    print("\n" + "".join(["*"] * 10) + f" Scoring a fefRBM model " + "".join(["*"] * 10) + "\n")
    device = get_device(args.device)
    print("\n")
    print(f"Input data:\t\t{args.data}")
    print(f"Input annotations:\t{args.annotations}")
    print(f"Input model:\t\t{args.path_params}")
    print(f"Output folder:\t\t{args.output}")
    print(f"Generation time:\t{args.gen_time}")
    print(f"Data type:\t\t{args.dtype}")
    print("\n")
    
    # Import data
    print("Importing dataset...")
    dataset = get_dataset(
        path_data=args.data,
        path_ann=args.annotations,
        path_weights=args.weights,
        alphabet=args.alphabet,
        device=device,
        dtype=args.dtype,
    )
    tokens = get_tokens(dataset.alphabet)
    print(f"Alphabet: {dataset.alphabet}")
    
    # Create the folder where to save the model
    folder = Path(args.output)
    folder.mkdir(parents=True, exist_ok=True)
    
    if args.label is not None:
        file_scores = folder / Path(f"{args.label}_scores.h5")        
    else:
        file_scores = folder / Path("scores.h5")
        
    # Delete the file if it already exists
    if file_scores.exists():
        file_scores.unlink()
        
    # Set the random seed
    torch.manual_seed(args.seed)
        
    # Shuffle the dataset
    dataset.shuffle()
    
    # Initialize the model
    num_visibles = dataset.get_num_residues()
    num_labels = dataset.get_num_classes()
    num_states = dataset.get_num_states()
    if isinstance(dataset, DatasetBin):
        is_categorical = False
        rbm = fefRBMbin()
        data = dataset.data
    elif isinstance(dataset, DatasetCat):
        is_categorical = True
        rbm = fefRBMcat()
        data = dataset.data_one_hot
    labels = dataset.labels_one_hot
    num_samples = len(data)
    
    # Saved checkpoints
    model_checkpoints = get_saved_updates(args.path_params)
    print('\nSaved checkpoints:')
    print(model_checkpoints)
    chpts = list(map(int, input('\nInsert the list of checkpoints you want to study (separated by commas): ').split(',')))
    # checking that the ages are present among the saved models
    for chpt in chpts:
        if chpt not in model_checkpoints:
            raise KeyError(chpt, 'is not among the saved checkpoints')
    
    # Computing the observables of the data
    print("\nComputing the observables of the data...")
    if is_categorical:
        fi_data = get_freq_single_point_cat(data, weights=dataset.weights, pseudocount=1e-8)
        fij_data = get_freq_two_points_cat(data, weights=dataset.weights, pseudocount=1e-8)
        cov_data = fij_data - torch.einsum('ij,kl->ijkl', fi_data, fi_data)
    else:
        fi_data = get_freq_single_point_bin(data, weights=dataset.weights, pseudocount=1e-8)
        fij_data = get_freq_two_points_bin(data, weights=dataset.weights, pseudocount=1e-8)
        cov_data = fij_data - torch.einsum('i,j->ij', fi_data, fi_data)
    spectrum_data = torch.linalg.svdvals(data.view(num_samples, -1) - data.view(num_samples, -1).mean(0)).square() / num_samples
    entropy_data = len(gzip.compress(data.int().cpu().numpy())) / num_samples
    
    # Define the generation times at which evaluating the model
    exponent = int(np.log10(args.gen_time))
    record_times = np.unique(np.logspace(0, exponent, args.num_checkpoints)).int()
    record_times = np.unique(np.sort(np.append([0, 10], record_times))) # put 0 and 10 as first two elements
    # For tracking all the generation process
    if args.num_checkpoints == args.gen_time:
        record_times = np.arange(0, args.num_checkpoints + 1)
    
    score_checkpoints = []
    for checkpoint in chpts:
        
        scores = {
            "log_likelihood" : [],
            "spectrum" : [],
            "entropy" : [],
            "AAI_data" : [],
            "AAI_gen" : [],
            "first_moment" : [],
            "second_moment" : [],
        }
        
        rbm.load(
            filename=args.path_params,
            index=checkpoint,
            device=device,
            dtype=dtype,
        )
        chains = rbm.init_chains(num_samples=num_samples)
        
        for t in range(args.gen_time):
            chains = rbm.sample_given_label(
                gibbs_steps=1,
                chains=chains,
            )
            if t in record_times:
                scores["log_likelihood"].append(scores.LL_score(rbm, data, chains["visible"], labels))
                scores["spectrum"].append(scores.spectrum_score(spectrum_data, chains["visible"]))
    
    