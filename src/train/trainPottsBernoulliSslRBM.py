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
from torch.nn.functional import one_hot
from torch.utils.data import Dataset
import numpy as np
from h5py import File
from pathlib import Path

# dataset class for this RBM

class RBMdataset(Dataset):
    def __init__(self, file_path, dataset='train', partial_labels=True, lab_frac=0.5):
        f = File(file_path, 'r')
        self.file_path = file_path
        self.data = torch.tensor(f[dataset][()]).type(torch.int64)
        self.partial_labels = partial_labels
        self.lab_frac = lab_frac
        labels_string = f[dataset + '_labels'].asstr()[()].flatten()
        label2category = {}
        self.categories = np.unique(labels_string)[np.unique(labels_string) != '-1']
        label2category = {lab : i for i, lab in enumerate(self.categories)}
        label2category['-1'] = -1
        all_labels = torch.tensor([label2category[lab] for lab in labels_string]).type(torch.int64)
        if self.partial_labels:
            self.labels = all_labels
        else:
            missing_labels = torch.ones(len(self.data), dtype=torch.int64) * -1
            n_label = int(len(all_labels) * lab_frac)
            label_idx = np.random.choice(np.arange(len(all_labels)), n_label, replace=False)
            missing_labels[label_idx] = all_labels[label_idx]
            self.labels = missing_labels
        f.close()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample_data = self.data[idx]
        sample_labels = self.labels[idx]
        return sample_data, sample_labels
    
    def get_num_visibles(self):
        return self.data.shape[1]
    
    def get_dataset_mean(self):
        return torch.mean(self.data.type(torch.float32), 0)
    
    def get_num_categs(self):
        return len(self.categories)
    
    def get_labels_frac(self):
        if self.partial_labels:
            missing_ratio = torch.sum(self.labels == -1)
            return float(1. - missing_ratio)
        else:
            return self.lab_frac
            
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
    num_states = torch.max(X) + 1
    all_states = torch.arange(num_states).reshape(-1,1,1)
    freq = (X == all_states).type(torch.float32).mean(1)
    freq = torch.clamp(freq, min=eps, max=1. - eps)
    
    return (torch.log(freq) - 1/num_states * torch.sum(torch.log(freq), 0))

def initLabBias(dataset : torch.utils.data.Dataset) -> torch.Tensor:
    X = dataset.labels
    eps = 1e-7
    num_states = torch.max(X) + 1
    freq = one_hot(X).type(torch.float32).mean(0)
    freq = torch.clamp(freq, min=eps, max=1. - eps)
    
    return torch.log(freq) - 1. / num_states * torch.sum(torch.log(freq), 0)
                          
# import command-line input arguments
parser = argparse.ArgumentParser(description='Train in semi-supervised mode an RBM with Potts varables in the visible layer and binary variables in the hidden layer.')
parser.add_argument('--data', '-d',         type=Path,  required=True,      help='Filename of the dataset to be used for training the model.')
parser.add_argument('--train_mode',         type=str,   default='new',      help='(Defaults to new). Wheather to start a new training or recover a new one.', choices=['new', 'restore'])
parser.add_argument('--num_states',         type=int,   default=21,         help='(Defaults to 21). Number of states of the Potts variables.')
parser.add_argument('--partial_labels',                 default=False,      help='(Defaults to False). If only partial labels are provided in the training dataset.', action='store_true')
parser.add_argument('--lab_frac',           type=float, default=1.0,        help='(Defaults to 1.0). Ratio of labels to keep. Used only if `partial_labels` is False.')
parser.add_argument('--train_type',         type=str,   default='PCD',      help='(Defaults to PCD). How to perform the training.', choices=['PCD', 'CD', 'Rdm'])
parser.add_argument('--n_save',             type=int,   default=20,         help='(Defaults to 20). Number of models to save during the training.')
parser.add_argument('--epochs',             type=int,   default=10000,      help='(Defaults to 10000). Number of epochs.')
parser.add_argument('--Nh',                 type=int,   default=1024,       help='(Defaults to 1024). Number of hidden units.')
parser.add_argument('--lr',                 type=float, default=0.01,       help='(Defaults to 0.01). Learning rate.')
parser.add_argument('--lr_labels',          type=float, default=0.01,       help='(Defaults to 0.01). Learning rate of the labels matrix.')
parser.add_argument('--l1_penalty',         type=float, default=0.,         help='(Defaults to 0.). L1 regularization parameter.')
parser.add_argument('--l2_penalty',         type=float, default=0.,         help='(Defaults to 0.). L2 regularization parameter.')
parser.add_argument('--n_gibbs',            type=int,   default=100,         help='(Defaults to 100). Number of Gibbs steps for each gradient estimation.')
parser.add_argument('--minibatch_size',     type=int,   default=500,        help='(Defaults to 500). Minibatch size.')
parser.add_argument('--n_chains',           type=int,   default=500,        help='(Defaults to 500). Number of permanent chains.')
parser.add_argument('--spacing',            type=str,   default='exp',      help='(Defaults to exp). Spacing to save models.', choices=['exp', 'linear'])
parser.add_argument('--no_center_gradient',             default=True,       help='Use this option if you don\'t want the centered gradient.', action='store_false')
parser.add_argument('--seed',               type=int,   default=0,          help='(Defaults to 0). Random seed.')
args = parser.parse_args()

# import the proper RBM class
RBM = importlib.import_module('PottsBernoulliSslRBM').RBM

device = utilities.select_device()

#####################################################################################################

# start a new training
if args.train_mode == 'new':

    # initialize random states
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Import data and RBM model
    train_dataset = RBMdataset(args.data, dataset='train', partial_labels=args.partial_labels, lab_frac=args.lab_frac)
    
    Nv = train_dataset.get_num_visibles()
    rbm = RBM(num_visible=Nv, num_hidden=args.Nh, num_states=args.num_states, num_categ=train_dataset.get_num_categs(), device=device)
    rbm.dataset_filename = str(args.data)
    rbm.seed = args.seed

    fname_out = rbm.generate_model_stamp(dataset_stamp=args.data.stem,
                                           epochs=args.epochs,
                                           learning_rate=args.lr,
                                           learning_rate_labels=args.lr_labels,
                                           gibbs_steps=args.n_gibbs,
                                           batch_size=args.minibatch_size,
                                           partial_labels=args.partial_labels,
                                           perc_labels=args.lab_frac,
                                           L1_reg=args.l1_penalty,
                                           L2_reg=args.l2_penalty,
                                           training_mode=args.train_type)
    
    model_folder = Path('models/' + args.data.stem)
    model_folder.mkdir(exist_ok=True)
    rbm.file_stamp = fname_out
    
    # Initialize the visible and label biases.
    rbm.vbias = initVisBias(train_dataset).to(device)
    rbm.lbias = initLabBias(train_dataset).to(device)

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
              lr_labels=args.lr_labels,
              batch_size=args.minibatch_size,
              gibbs_steps=args.n_gibbs,
              updCentered=args.no_center_gradient,
              L1_reg=args.l1_penalty,
              L2_reg=args.l2_penalty
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
    rbm = RBM(num_visible=0, num_hidden=0, device=device)
    rbm.loadRBM(model_path)

    # load data
    f_data = File(rbm.dataset_filename, 'r')
    train_dataset = RBMdataset(rbm.dataset_filename, dataset='train', partial_labels=rbm.partial_labels, lab_frac=rbm.lab_frac)
    f_data.close()

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