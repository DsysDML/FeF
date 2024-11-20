from pathlib import Path
import argparse
import numpy as np
from tqdm import tqdm
from itertools import cycle
import time
import warnings

import torch
from torch.optim import SGD
from adabmDCA.utils import get_device
from adabmDCA.fasta_utils import get_tokens
from annadca.dataset import DatasetBin, DatasetCat, get_dataset
from annadca.dataset import DataLoader_shuffle as DataLoader
from annadca.utils import get_saved_updates

from fef.parser import add_args_train
from fef.rbms import fefRBMbin
from fef.rbms import fefRBMcat
from fef.train import fef


# import command-line input arguments
def create_parser():
    # Important arguments
    parser = argparse.ArgumentParser(description='Training a fefRBM model')
    parser = add_args_train(parser) 
    
    return parser


if __name__ == '__main__':
    
    # Load parser, training dataset and DCA model
    parser = create_parser()
    args = parser.parse_args()
    
    if args.dtype == "float32":
        args.dtype = torch.float32
    elif args.dtype == "float64":
        args.dtype = torch.float64
    
    print("\n" + "".join(["*"] * 10) + f" Training fefRBM model " + "".join(["*"] * 10) + "\n")
    device = get_device(args.device)
    print("\n")
    print(f"Input data:\t\t{args.data}")
    print(f"Output folder:\t\t{args.output}")
    print(f"Number of hidden units:\t{args.hidden}")
    print(f"Learning rate:\t\t{args.lr}")
    print(f"Minibatch size:\t\t{args.nchains}")
    print(f"Number of chains:\t{args.nchains}")
    print(f"Number of Gibbs Steps:\t{args.gibbs_steps}")
    print(f"Number of epochs:\t{args.nepochs}")
    print(f"Centered gradient:\t{not args.uncentered}")
    print(f"Labels contribution:\t{args.eta}")
    print(f"Use single gradient:\t{args.single_gradient}")
    if args.pseudocount is not None:
        print(f"Pseudocount:\t\t{args.pseudocount}")
    print(f"Random seed:\t\t{args.seed}")
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
        file_paths = {
            "log" : folder / Path(f"{args.label}.log"),
            "params" : folder / Path(f"{args.label}_params.h5"),
        }
        
    else:
        file_paths = {
            "log" : folder / Path(f"adabmDCA.log"),
            "params" : folder / Path(f"params.h5"),
        }
        
    # Check if the files in file_paths already exist. If so, delete them
    for path in file_paths.values():
        if path.exists():
            path.unlink()
            
    # Save the weights if not already provided
    if args.weights is None:
        if args.label is not None:
            path_weights = folder / f"{args.label}_weights.dat"
        else:
            path_weights = folder / "weights.dat"
        np.savetxt(path_weights, dataset.weights.cpu().numpy())
        print(f"Weights saved in {path_weights}")
        
    # Set the random seed
    torch.manual_seed(args.seed)
        
    # Shuffle the dataset
    dataset.shuffle()
    
    if args.pseudocount is None:
        args.pseudocount = 1. / dataset.get_effective_size()
        print(f"Pseudocount automatically set to {args.pseudocount}.")
        
    # Initialize the model
    num_visibles = dataset.get_num_residues()
    num_hiddens = args.hidden
    num_labels = dataset.get_num_classes()
    num_states = dataset.get_num_states()
    if isinstance(dataset, DatasetBin):
        rbm = fefRBMbin()
        data = dataset.data
    elif isinstance(dataset, DatasetCat):
        rbm = fefRBMcat()
        data = dataset.data_one_hot
    
    if args.path_params is not None:
        rbm.load(
            filename=args.path_params,
            device=device,
            dtype=args.dtype,
        )
        
    else:            
        rbm.init_parameters(
            num_visibles=num_visibles,
            num_hiddens=num_hiddens,
            num_labels=num_labels,
            num_states=num_states,
            frequencies_visibles=None,
            frequencies_labels=None,
            std_init=1e-4,
            device=device,
            dtype=args.dtype,
        )
    
    if args.nchains >= dataset.__len__():
        args.nchains = dataset.__len__()
        warnings.warn("The number of chains is larger than the dataset size. The number of chains is set to the dataset size.")
        
    print("\n")
    # Save the hyperparameters of the model
    template = "{0:20} {1:10}\n"  
    with open(file_paths["log"], "w") as f:
        if args.label is not None:
            f.write(template.format("label:", args.label))
        else:
            f.write(template.format("label:", "N/A"))
            
        f.write(template.format("input data:", str(args.data)))
        f.write(template.format("alphabet:", dataset.alphabet))
        f.write(template.format("# hiddens:", args.hidden))
        f.write(template.format("nchains:", args.nchains))
        f.write(template.format("minibatch size:", args.nchains))
        f.write(template.format("gibbs steps:", args.gibbs_steps))
        f.write(template.format("lr:", args.lr))
        f.write(template.format("pseudo count:", args.pseudocount))
        f.write(template.format("centered:", not args.uncentered))
        f.write(template.format("single gradient:", args.single_gradient))
        f.write(template.format("eta:", args.eta))
        f.write(template.format("random seed:", args.seed))
        f.write("\n")
        template = "{0:10} {1:10}\n"
        f.write(template.format("Epoch", "Time [s]"))
        
    # Initialize gradients for the parameters
    for key, value in rbm.params.items():
        value.grad = torch.zeros_like(value)

    # Select the optimizer
    optimizer = SGD(rbm.params.values(), lr=args.lr, maximize=True)
    optimizer.zero_grad(set_to_none=False)
    
    # Initialize the dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=args.nchains,
        shuffle=True,
        drop_last=True,
    )
    
    # Allows to iterate indefinitely on the dataloader without worrying on the epochs
    dataloader = cycle(dataloader)
    
    # Train the model
    start = time.time()
    pbar = tqdm(initial=0, total=args.nepochs, colour="red", dynamic_ncols=True, ascii="-#")
    upd = 0
    with torch.no_grad():
        while upd < args.nepochs:
            
            upd += 1
            if upd % 10 == 0:
                pbar.update(10)

            # Get the next batch
            batch = next(dataloader)
            fef(
                rbm=rbm,
                data_batch=batch,
                gibbs_steps=args.gibbs_steps,
                optimizer=optimizer,
                pseudo_count=args.pseudocount,
                centered=(not args.uncentered),
                eta=args.eta,
                single_gradient=args.single_gradient,
            )
            
            # Set the gauge of the weights
            rbm.zerosum_gauge()

            if upd % 1000 == 0:
                rbm.save(
                    filename=file_paths["params"],
                    num_updates=upd,
                )
                with open(file_paths["log"], "a") as f:
                    f.write(template.format(f"{upd}", f"{time.time() - start:.2f}"))
        pbar.close()
        
        # Save the final model if upd is not present in the h5 archive
        saved_updates = get_saved_updates(file_paths["params"])
        if upd not in saved_updates:
            rbm.save(
                filename=file_paths["params"],
                num_updates=upd,
            )
            with open(file_paths["log"], "a") as f:
                f.write(template.format(f"{upd}", f"{time.time() - start:.2f}"))