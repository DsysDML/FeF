#!/usr/bin/python3

import sys
import os
if os.getenv('RBMHOME') != None:
    os.chdir(os.getenv('RBMHOME'))
sys.path.append(os.getcwd() + '/src')
sys.path.append(os.getcwd() + '/src/RBMs')
import importlib
import argparse
import utilities
import torch
from torch.utils.data import Dataset
import numpy as np
from h5py import File
from pathlib import Path

# dataset class for this RBM

class RBMdataset(Dataset):
    def __init__(self, file_path, dataset='train', dtype=torch.float32):
        f = File(file_path, 'r')
        self.file_path = file_path
        self.dtype = dtype
        self.data = torch.tensor(f[dataset][()]).type(dtype)
        f.close()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return sample
    
    def get_num_visibles(self):
        return self.data.shape[1]
    
    def get_dataset_mean(self):
        return torch.mean(self.data.type(self.dtype), 0)
    
    def is_continuous(self):
        # returns true if the dataset is made of continuous variables. This determines the type of batch
        # initialization if Rdm training mode is used
        if torch.all(torch.remainder(self.data, 1) == 0):
            return False
        else:
            return True
    
def initVisBias(dataset : torch.utils.data.Dataset) -> torch.Tensor:
    """Initialize the visible biases by minimizing the distance with the independent model.

    Args:
        dataset (torch.utils.data.Dataset): Dataset.
        device (torch.device, optional): Device. Defaults to torch.device('cpu').

    Returns:
        torch.Tensor: Visible biases of the independent variables model.
    """
    
    X = dataset.data
    eps = 1e-7
    freq = X.mean(0)
    freq = torch.clamp(freq, min=eps, max=1. - eps)
    
    return torch.log(freq) - torch.log(1. - freq)
                          
# import command-line input arguments
parser = argparse.ArgumentParser(description='Train an RBM with binary variables both in the visible and in the hidden layer.')
parser.add_argument('--data', '-d',         type=Path,  required=True,      help='Filename of the dataset to be used for training the model.')
parser.add_argument('--train_mode',         type=str,   default='new',      help='(Defaults to new). Wheather to start a new training or recover a new one.', choices=['new', 'restore'])
parser.add_argument('--train_type',         type=str,   default='PCD',      help='(Defaults to PCD). How to perform the training.', choices=['PCD', 'CD', 'Rdm'])
parser.add_argument('--n_save',             type=int,   default=20,        help='(Defaults to 20). Number of models to save during the training.')
parser.add_argument('--epochs',             type=int,   default=1000,      help='(Defaults to 1000). Number of epochs.')
parser.add_argument('--Nh',                 type=int,   default=100,        help='(Defaults to 100). Number of hidden units.')
parser.add_argument('--lr',                 type=float, default=0.01,       help='(Defaults to 0.01). Learning rate.')
parser.add_argument('--n_gibbs',            type=int,   default=50,        help='(Defaults to 50). Number of Gibbs steps for each gradient estimation.')
parser.add_argument('--minibatch_size',     type=int,   default=500,        help='(Defaults to 500). Minibatch size.')
parser.add_argument('--n_chains',           type=int,   default=500,        help='(Defaults to 500). Number of permanent chains.')
parser.add_argument('--spacing',            type=str,   default='exp',   help='(Defaults to exp). Spacing to save models.', choices=['exp', 'linear'])
parser.add_argument('--no_center_gradient',             default=True,       help='Use this option if you don\'t want the centered gradient.', action='store_false')
parser.add_argument('--seed',               type=int,   default=0,          help='(Defaults to 0). Random seed.')
args = parser.parse_args()

dtype = torch.float32

# import the proper RBM class
RBM = importlib.import_module('BernoulliBernoulliRBM').RBM

device = utilities.select_device()

#####################################################################################################

# start a new training
if args.train_mode == 'new':

    # initialize random states
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Import data and RBM model
    train_dataset = RBMdataset(args.data, dataset='train', dtype=dtype)
    Nv = train_dataset.get_num_visibles()
    rbm = RBM(num_visible=Nv, num_hidden=args.Nh, device=device, dtype=dtype)
    rbm.dataset_filename = str(args.data)
    rbm.seed = args.seed
    rbm.continuous_dataset = train_dataset.is_continuous()

    fname_out = rbm.generate_model_stamp(dataset_stamp=args.data.stem,
                                           epochs=args.epochs,
                                           learning_rate=args.lr,
                                           gibbs_steps=args.n_gibbs,
                                           batch_size=args.minibatch_size,
                                           training_mode=args.train_type)
    
    model_folder = Path('models/' + args.data.stem)
    model_folder.mkdir(exist_ok=True)
    rbm.file_stamp = fname_out

    # Initialize the visible biases.
    rbm.vbias = initVisBias(train_dataset).to(device)

    # Check if the file already exists. If so, ask wheather to overwrite it or not.
    utilities.check_if_exists(fname_out)

    # Select the list of training times (ages) at which saving the model.
    if args.spacing == 'exp':
        list_save_rbm = []
        xi = args.epochs
        for i in range(args.n_save):
            list_save_rbm.append(xi)
            xi = xi / args.epochs ** (1 / args.n_save)

        list_save_rbm = np.unique(np.array(list_save_rbm, dtype=np.int32))
    
    elif args.spacing == 'linear':
        list_save_rbm = np.linspace(1, args.epochs, args.n_save).astype(np.int32)

    rbm.list_save_rbm = list_save_rbm

    # fit the model
    rbm.fit(train_dataset,
              training_mode=args.train_type,
              epochs=args.epochs,
              num_pcd=args.n_chains,
              lr=args.lr,
              batch_size=args.minibatch_size,
              gibbs_steps=args.n_gibbs,
              updCentered=args.no_center_gradient
             )

    print('\nTraining time: {:.1f} minutes'.format(rbm.training_time / 60))

#################################################################################################

# restore an old training session
elif args.train_mode == 'restore':

    model_path = utilities.catch_file('models', message='Insert the path of the existing model: ')
    f_rbm = File(model_path, 'a')

    # restore the random states
    torch.set_rng_state(torch.tensor(np.array(f_rbm['torch_rng_state'])))
    np_rng_state = tuple([f_rbm['numpy_rng_arg0'][()].decode('utf-8'),
                            f_rbm['numpy_rng_arg1'][()],
                            f_rbm['numpy_rng_arg2'][()],
                            f_rbm['numpy_rng_arg3'][()],
                            f_rbm['numpy_rng_arg4'][()]])
    np.random.set_state(np_rng_state)
    
    # load RBM
    rbm = RBM(num_visible=0, num_hidden=0, device=device, dtype=dtype)
    rbm.loadRBM(model_path)

    # load data
    f_data = File(rbm.dataset_filename, 'r')
    train_dataset = RBMdataset(rbm.dataset_filename, dataset='train', dtype=dtype)
    f_data.close()
    rbm.continuous_dataset = train_dataset.is_continuous()

    ep_max_old = rbm.ep_tot
    print('This model has been trained until age: ', ep_max_old, 'epochs')
    ep_max_new = int(input('Insert the new age (epochs): ')) # put the same t_age of the previous RBM to recover an interrupted training

    rbm.ep_start = ep_max_old + 1
    rbm.ep_max = ep_max_new
    
    if ep_max_new > rbm.list_save_rbm[-1]:
        if args.spacing == 'exp':
            dlog = np.log(rbm.list_save_rbm[-2] / rbm.list_save_rbm[-3]) # previous log spacing
            new_saving_points = np.exp(np.arange(np.log(ep_max_old) + dlog, np.log(ep_max_new), dlog)).astype(np.int64)
            new_saving_points = np.append(new_saving_points - 1, ep_max_new)
        elif args.spacing == 'linear':
            dlin = rbm.list_save_rbm[-2] - rbm.list_save_rbm[-3]
            new_saving_points = np.arange(ep_max_old + dlin, ep_max_new, dlin).astype(np.int64)
            new_saving_points = np.append(new_saving_points, [ep_max_new])
        rbm.list_save_rbm = np.append(rbm.list_save_rbm, new_saving_points)

    rbm.fit(train_dataset, restore=True)
    f_rbm.close()
    print('\nTraining time: {:.1f} minutes'.format(rbm.training_time / 60))