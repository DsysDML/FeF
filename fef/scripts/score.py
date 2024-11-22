import argparse
from pathlib import Path
from sklearn.metrics import confusion_matrix
import gzip
import numpy as np
import h5py
import matplotlib.pyplot as plt
import torch

from adabmDCA.utils import get_device
from adabmDCA.fasta_utils import get_tokens
from adabmDCA.stats import get_freq_single_point as get_freq_single_point_cat
from adabmDCA.stats import get_freq_two_points as get_freq_two_points_cat
from annadca.binary.stats import get_freq_single_point as get_freq_single_point_bin
from annadca.binary.stats import get_freq_two_points as get_freq_two_points_bin
from annadca.utils import get_eigenvalues_history
from annadca.utils import get_saved_updates
from annadca.dataset import DatasetBin, DatasetCat, get_dataset

from fef.rbms import fefRBMbin, fefRBMcat
from fef.scores import Score
import fef.scores as scores
from fef.parser import add_args_score
from fef.plot import (
    plot_scores_mean,
    plot_eigenvalues,
    plot_AAI_scores,
    plot_accuracies,
    plot_confusion_matrix,
)


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
    
    if args.use_latex:
        plt.rcParams['mathtext.fontset'] = 'stix'
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams.update({'font.size': 15})
        plt.rcParams['text.usetex'] = True
        plt.rcParams['text.latex.preamble'] = r'\usepackage{bm}'
        plt.rcParams['legend.title_fontsize'] = 'xx-small'
    
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
        dtype=dtype,
    )
    tokens = get_tokens(dataset.alphabet)
    print(f"Alphabet: {dataset.alphabet}")
    
    # Create the folder where to save the model
    folder = Path(args.output)
    folder.mkdir(parents=True, exist_ok=True)
    
    folder_scores = folder / Path("scores")
    folder_scores.mkdir(parents=True, exist_ok=True)
    folder_plots = folder / Path("plots")
    folder_plots.mkdir(parents=True, exist_ok=True)
        
    # Set the random seed
    torch.manual_seed(args.seed)
        
    # Shuffle the dataset
    dataset.shuffle()
    
    # Get the labels
    targets = dataset.to_label(dataset.labels_one_hot)
    unique_labels = np.unique(targets)
    
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
    labels_one_hot = dataset.labels_one_hot
    num_samples = len(data)
    
    # Saved checkpoints
    model_checkpoints = get_saved_updates(args.path_params)
    if args.checkpoints is not None:
        chpts = list(map(int, args.checkpoints.split(',')))
        for chpt in chpts:
            if chpt not in model_checkpoints:
                raise KeyError(chpt, 'is not among the saved checkpoints')
    else:
        print('\nSaved checkpoints:')
        print(model_checkpoints)
        chpts = list(map(int, input('\nInsert the list of checkpoints you want to study (separated by commas): ').split(',')))
        # checking that the ages are present among the saved models
        for chpt in chpts:
            if chpt not in model_checkpoints:
                raise KeyError(chpt, 'is not among the saved checkpoints')
    
    # Computing the observables of the data
    print("\nComputing the observables of the data...")
    data_observables = {label: {} for label in unique_labels}
    if is_categorical:
        for label in unique_labels:
            mask = targets == label
            fi_data = get_freq_single_point_cat(data[mask], weights=dataset.weights[mask], pseudo_count=1e-8)
            fij_data = get_freq_two_points_cat(data[mask], weights=dataset.weights[mask], pseudo_count=1e-8)
            cov_data = fij_data - torch.einsum('ij,kl->ijkl', fi_data, fi_data)
            data_observables[label]["fi"] = fi_data
            data_observables[label]["cov"] = cov_data
    else:
        for label in unique_labels:
            mask = targets == label
            fi_data = get_freq_single_point_bin(data[mask], weights=dataset.weights[mask], pseudo_count=1e-8)
            fij_data = get_freq_two_points_bin(data[mask], weights=dataset.weights[mask], pseudo_count=1e-8)
            cov_data = fij_data - torch.einsum('i,j->ij', fi_data, fi_data)
            data_observables[label]["fi"] = fi_data
            data_observables[label]["cov"] = cov_data
    for label in unique_labels:
        mask = targets == label
        data_observables[label]["spectrum"] = torch.linalg.svdvals(data[mask].view(mask.sum(), -1) - data[mask].view(mask.sum(), -1).mean(0)).square() / mask.sum()
        data_observables[label]["entropy"] = len(gzip.compress(data[mask].int().cpu().numpy())) / mask.sum()
    
    # Define the generation times at which evaluating the model
    exponent = int(np.log10(args.gen_time))
    record_times = np.unique(np.logspace(0, exponent, args.num_records)).astype(int)
    record_times = np.unique(np.sort(np.append([1, 10], record_times))) # put 0 and 10 as first two elements
    # For tracking all the generation process
    if args.num_records == args.gen_time:
        record_times = np.arange(0, args.num_records + 1)
    
    # Dummy functions to split AAI_data from AAI_gen when computing scores.AAI_score
    def get_AAI_data(AAI_scores: tuple) -> float:
        return AAI_scores[0]
    def get_AAI_gen(AAI_scores: tuple) -> float:
        return AAI_scores[1]
    # Function for computing the accuracy in predicting the labels
    def get_accuracy(target: torch.Tensor, prediction: torch.Tensor) -> float:
        return (target == prediction).sum().item() / len(target)
    
    # Initialize the Score objects
    score_ll = Score(checkpoints=chpts, record_times=record_times, labels=unique_labels, score_function=scores.LL_score)
    score_spectrum = Score(checkpoints=chpts, record_times=record_times, labels=unique_labels, score_function=scores.spectrum_score)
    score_entropy = Score(checkpoints=chpts, record_times=record_times, labels=unique_labels, score_function=scores.entropy_score)
    score_AAI_data = Score(checkpoints=chpts, record_times=record_times, labels=unique_labels, score_function=get_AAI_data)
    score_AAI_gen = Score(checkpoints=chpts, record_times=record_times, labels=unique_labels, score_function=get_AAI_gen)
    score_first_moment = Score(checkpoints=chpts, record_times=record_times, labels=unique_labels, score_function=scores.first_moment_score)
    score_second_moment = Score(checkpoints=chpts, record_times=record_times, labels=unique_labels, score_function=scores.second_moment_score)
    score_accuracy = Score(checkpoints=chpts, record_times=record_times, labels=unique_labels, score_function=get_accuracy)
    confusion_matrices = {}
    
    for checkpoint in chpts:
        print(f"\nEvaluating the model at checkpoint {checkpoint}...")
        
        # Load the model
        rbm.load(
            filename=args.path_params,
            index=checkpoint,
            device=device,
            dtype=dtype,
        )
        chains_gen = rbm.init_chains(num_samples=num_samples)
        chains_gen["label"] = labels_one_hot
        chains_predict = rbm.init_chains(num_samples=num_samples)
        chains_predict["visible"] = data
        
        for t in range(1, args.gen_time + 1):
            chains_gen = rbm.sample_given_label(
                gibbs_steps=1,
                chains=chains_gen,
            )
            chains_predict = rbm.sample_given_visible(
                gibbs_steps=1,
                chains=chains_predict,
            )
            if t in record_times:
                print(f"Record time: {t} of {args.gen_time}")
                for label in unique_labels:
                    mask = targets == label
                    score_ll.evaluate(label=label, checkpoint=checkpoint, record_time=t, rbm=rbm, data=data[mask], gen=chains_gen["visible"][mask], labels=chains_gen["label"][mask])
                    score_spectrum.evaluate(label=label, checkpoint=checkpoint, record_time=t, spectrum_data=data_observables[label]["spectrum"], gen=chains_gen["visible"][mask])
                    score_entropy.evaluate(label=label, checkpoint=checkpoint, record_time=t, entropy_data=data_observables[label]["entropy"], gen=chains_gen["visible"][mask])
                    score_first_moment.evaluate(label=label, checkpoint=checkpoint, record_time=t, fi_data=data_observables[label]["fi"], gen=chains_gen["visible"][mask])
                    score_second_moment.evaluate(label=label, checkpoint=checkpoint, record_time=t, cov_data=data_observables[label]["cov"], gen=chains_gen["visible"][mask])
                    scores_AAI = scores.AAI_score(data=data[mask], gen=chains_gen["visible"][mask])
                    score_AAI_data.evaluate(label=label, checkpoint=checkpoint, record_time=t, AAI_scores=scores_AAI)
                    score_AAI_gen.evaluate(label=label, checkpoint=checkpoint, record_time=t, AAI_scores=scores_AAI)
                    score_accuracy.evaluate(label=label, checkpoint=checkpoint, record_time=t, target=dataset.to_label(labels_one_hot[mask]), prediction=dataset.to_label(chains_predict["label"][mask]))
        predictions = dataset.to_label(chains_predict["label"].cpu().numpy())
        confusion_matrices[checkpoint] = confusion_matrix(targets, predictions, normalize="true")
    # Save the scores
    score_ll.save(folder_scores / Path("log_likelihood.h5"))
    score_spectrum.save(folder_scores / Path("spectrum.h5"))
    score_entropy.save(folder_scores / Path("entropy.h5"))
    score_first_moment.save(folder_scores / Path("first_moment.h5"))
    score_second_moment.save(folder_scores / Path("second_moment.h5"))
    score_AAI_data.save(folder_scores / Path("AAI_data.h5"))
    score_AAI_gen.save(folder_scores / Path("AAI_gen.h5"))
    score_accuracy.save(folder_scores / Path("accuracy.h5"))
    with h5py.File(folder_scores / Path("confusion_matrices.h5"), "w") as f:
        for checkpoint, confusion_matrix in confusion_matrices.items():
            f.create_dataset(str(checkpoint), data=confusion_matrix)
    
    # Plot the scores mean for the generation
    plot_scores_mean(
        filename=folder_plots / Path("scores_mean.png"),
        score_ll=score_ll,
        score_spectrum=score_spectrum,
        score_entropy=score_entropy,
        score_first_moment=score_first_moment,
        score_second_moment=score_second_moment,
        score_AAI_data=score_AAI_data,
        score_AAI_gen=score_AAI_gen,
    )
    
    # Compute and plot the eigenvalues of the weight matrix and label matrix
    updates, eigenvalues = get_eigenvalues_history(args.path_params, target_matrix="weight_matrix", device=device, dtype=dtype)
    updates, eigenvalues_labels = get_eigenvalues_history(args.path_params, target_matrix="label_matrix", device=device, dtype=dtype)
    plot_eigenvalues(
        filename=folder_plots / Path("eigenvalues.png"),
        updates=updates,
        eigenvalues=eigenvalues,
        eigenvalues_labels=eigenvalues_labels,
    )
    
    # Plot the AAI scores
    plot_AAI_scores(
        filename=folder_plots / Path("AAI_scores.png"),
        score_AAI_data=score_AAI_data,
        score_AAI_gen=score_AAI_gen,
    )
    
    # Plot the accuracies
    plot_accuracies(
        filename=folder_plots / Path("accuracy.png"),
        score_accuracy=score_accuracy,
    )
    
    # Plot the confusion matrices
    path_confusion_matrices = folder_plots / Path("confusion_matrices")
    path_confusion_matrices.mkdir(parents=True, exist_ok=True)
    for checkpoint, confusion_matrix in confusion_matrices.items():
        plot_confusion_matrix(
            filename=path_confusion_matrices / Path(f"confusion_matrix_{checkpoint}.png"),
            confusion_matrix=confusion_matrix,
            checkpoint=checkpoint,
            labels=unique_labels,
        )
    
            
    
    