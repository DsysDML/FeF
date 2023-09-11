import torch
from torch.distributions import Categorical
from torch.utils.data import DataLoader
import h5py
import numpy as np
import os
import time
from tqdm import tqdm
import datetime

class RBM:

    def __init__(self,
                num_visible=0,                  # Number or visisble variables
                num_hidden=0,                   # Number of hidden variables
                device= torch.device('cpu'),    # CPU or GPU?
                num_states=21,                  # Number of Potts states
                var_init=1e-4,                  # Variance of the initial weights
                dtype=torch.float32):           # Data type used during computations
        
        # structure-specific variables
        self.Nv = num_visible        
        self.Nh = num_hidden
        self.num_states = num_states
        self.device = device
        self.dtype = dtype
        self.var_init = var_init
        self.seed = 0
        
        # training-specific variables
        self.gibbs_steps = 0
        self.num_pcd = 0
        self.lr = 0
        self.ep_max = 0
        self.ep_start = 0 # epoch to start with, used for restoring the training
        self.mb_s = 0
        self.training_mode = ''
        self.updCentered = True
        self.UpdByEpoch = 0
        self.ep_tot = 0
        self.up_tot = 0
        self.list_save_rbm = []        
        self.time_start = 0
        self.training_time = 0
        
        # weights of the RBM
        self.W = torch.randn(size=(self.num_states, self.Nv, self.Nh), device=self.device, dtype=self.dtype) * self.var_init
        # visible and hidden biases
        self.vbias = torch.zeros((self.num_states, self.Nv), device=self.device, dtype=self.dtype)
        self.hbias = torch.zeros(self.Nh, device=self.device, dtype=self.dtype)
        # permanent chain
        self.X_pc = None
        # averages used to center the gradient
        self.deltaDataAv = torch.tensor([0])
        self.hidDataAv = torch.tensor([0])
        
        # identity variables
        self.dataset_filename = ''
        timestamp = str('.'.join(list(str(time.localtime()[i]) for i in range(5))))
        self.model_stamp = f'PottsBernoulliRBM-{timestamp}'
        self.file_stamp = ''
        
        # auxiliary variables
        self.all_states = torch.arange(self.num_states, device=self.device).reshape(-1,1,1)
        self.all_v = torch.arange(self.Nv, device=self.device)
        
        # constants
        self.eps = 1e-7 # precision for clamping values
    
    def loadRBM(self, fname : str, stamp=None) -> None:
        """Loads the RBM saved in fname at t_age = stamp. If stamp is not provided, the oldest model is used.

        Args:
            fname (str): Path of the model to load.
            stamp (int, optional): Age of the RBM. If 'stamp' is not present among the saved models, the closest one is taken. Defaults to None.
        """

        f = h5py.File(fname, 'r')
        if stamp is not None:
            stamp = str(stamp)
        else:
            stamp = str(f['ep_tot'][()])

        # if stamp is not present, take the closest value
        alltime = f['alltime'][()]
        if int(stamp) not in alltime:
            idx = np.argmin(np.abs(alltime - int(stamp)))
            stamp = str(alltime[idx])
        self.t_age = stamp

        self.W = torch.tensor(np.array(f['W' + stamp]))
        self.vbias = torch.tensor(np.array(f['vbias' + stamp]))
        self.hbias = torch.tensor(np.array(f['hbias' + stamp])) 
        self.Nv = self.W.shape[1]
        self.Nh = self.W.shape[2]
        self.num_states = self.W.shape[0]
        self.all_v = torch.arange(self.Nv, device=self.device)
        self.all_states = torch.arange(self.num_states, device=self.device).reshape(-1,1,1)
        self.gibbs_steps = f['NGibbs'][()]
        self.var_init = f['var_init'][()]
        self.num_pcd = f['numPCD'][()]
        self.lr = f['lr'][()]
        self.ep_max = f['ep_max'][()]
        self.mb_s = f['miniBatchSize'][()]
        self.ep_tot = f['ep_tot'][()]
        self.up_tot = f['up_tot'][()]
        self.list_save_rbm = f['alltime'][()]
        self.file_stamp = f['file_stamp'][()].decode('utf-8')
        self.updCentered = f['updCentered'][()]
        self.UpdByEpoch = f['UpdByEpoch'][()]
        self.dataset_filename = f['dataset_filename'][()].decode('utf8')
        self.model_stamp = f['model_stamp'][()].decode('utf8')
        self.training_time = f['training_time'][()]
        self.time_start = f['time_start'][()]
        self.seed = f['seed'][()]
        
        if self.updCentered:
            self.deltaDataAv = torch.tensor(np.array(f['deltaDataAv'][()]), device=self.device)
            self.hidDataAv = torch.tensor(np.array(f['hidDataAv'][()]), device=self.device)
        
        self.training_mode = f['training_mode'][()].decode('utf-8')
        if self.training_mode == 'Rdm':
            self.CDLearning = False
            self.ResetPermChainBatch = True
        elif self.training_mode == 'PCD':
            self.CDLearning = False
            self.ResetPermChainBatch = False
            self.X_pc = torch.tensor(np.array(f['X_pc']))
        else:
            self.CDLearning = True
            self.ResetPermChainBatch = False # this has no actual effect with CD
        
        if self.device.type != 'cpu':
            self.W = self.W.to(self.device)
            self.vbias = self.vbias.to(self.device)
            self.hbias = self.hbias.to(self.device)
            if self.training_mode == 'PCD':
                self.X_pc = self.X_pc.to(self.device)
    
    def compute_energy(self, V : torch.Tensor, H : torch.Tensor) -> torch.Tensor:
        """Computes the Hamiltonian on the visible (V) and hidden (H) variables.

        Args:
            V (torch.Tensor): Visible units.
            H (torch.Tensor): Hidden units.

        Returns:
            torch.Tensor: Energy of the data points.
        """

        fields = self.vbias[V, self.all_v].sum(1) + torch.tensordot(self.hbias, H, dims=[[0], [1]])
        interaction = torch.bmm(self.W[V, self.all_v, :], H.unsqueeze(-1)).squeeze(-1).sum(1)

        return - fields - interaction
    
    def compute_energy_visibles(self, V : torch.Tensor) -> torch.Tensor:
        """Computes the Hamiltonian on the visible variables only.

        Args:
            V (torch.Tensor): Visible units.

        Returns:
            torch.Tensor: Energy of the data points.
        """

        field = self.vbias[V, self.all_v].sum(1)
        exponent = self.hbias + self.W[V, self.all_v, :].sum(1) # (Ns, Nh)
        idx = exponent < 10
        log_term = exponent.clone()
        log_term[idx] = torch.log(1 + torch.exp(exponent[idx]))
        energy = - field - log_term.sum(1)

        return energy
    
    def compute_partition_function_AIS(self, n_beta : int) -> float:
        """Estimates the partition function of the model using Annealed Importance Sampling.

        Args:
            n_beta (int): Number of inverse temperatures that define the trajectories.

        Returns:
            float: Estimate of the log-partition function.
        """
        
        n_chains = self.num_pcd
        E = torch.zeros(n_chains, device=self.device, dtype=torch.float64)
        beta_list = np.linspace(0, 1., n_beta)
        dB = 1. / n_beta
        # initialize the chains
        v = torch.randint(0, self.num_states, (1, self.Nv), device=self.device).repeat(n_chains, 1)
        h = torch.randint(0, 2, (1, self.Nh), device=self.device).repeat(n_chains, 1).type(torch.float32)
        energy0 = torch.zeros(n_chains, device=self.device, dtype=torch.float64)
        energy1 = self.compute_energy(v, h).type(torch.float64)
        E += energy1 - energy0.type(torch.float64)
        for beta in beta_list:
            v, _ = self.sampleVisibles(h, beta=beta)
            h, _ = self.sampleHiddens(v, beta=beta)
            E += self.compute_energy(v, h).type(torch.float64)
            
        # Subtract the average for avoiding overflow
        W = (-dB * E).double()
        W_ave = W.mean()
        logZ0 = (self.Nv + self.Nh) * np.log(2)
        logZ = logZ0 + torch.log(torch.mean(torch.exp(W - W_ave))) + W_ave
        
        return logZ

    def sampleHiddens(self, V : torch.Tensor, beta=1.) -> list:
        """Samples the hidden variables by performing one block Gibbs sampling step.

        Args:
            V (torch.Tensor): Visible units.
            beta (float, optional): Inverse temperature. Defaults to 1.

        Returns:
            list: (hidden variables, hidden magnetizations)
        """

        # V is a batch of size (Ns, Nv)        
        I = self.W[V, self.all_v, :].sum(1) # (Ns, Nh)
        mh = torch.sigmoid(beta * (self.hbias + I))
        h = torch.bernoulli(mh)

        return h, mh

    def sampleVisibles(self, H : torch.Tensor, beta=1.) -> list:
        """Samples the visible variables by performing one block Gibbs sampling step.

        Args:
            H (torch.Tensor): Hidden units.
            beta (float, optional): Inverse temperature. Defaults to 1..

        Returns:
            list: (visible variables, visible magnetizations)
        """

        # H is a batch of size (Ns, Nh)
        z = self.vbias + torch.tensordot(H, self.W, dims=([1],[2])) # (Ns, num_states, Nv)
        mv = torch.softmax(beta * z.transpose(1, 2), 2) # (Ns, Nv, num_states)
        
        # sampling from a multinomial distribution
        v = Categorical(logits=(beta * z.transpose(1, 2))).sample() # (Ns, Nv)

        return v, mv
    
    def getAv(self) -> list:
        """Performs it_mcmc Gibbs steps. Used for the training.
        Returns:
            list: (visible variables, visible magnetizations, hidden variables, hidden magnetizations)
        """

        v = self.X_pc
        h, _ = self.sampleHiddens(v)
        v, _ = self.sampleVisibles(h)

        for _ in range(1, self.gibbs_steps):
            h, _ = self.sampleHiddens(v)
            v, _ = self.sampleVisibles(h)
        
        return v, h

    def sampling(self, X : torch.Tensor, it_mcmc : int=None, batch_size : int=-1) -> list:
        """Samples variables and magnetizatons starting from the initial condition X.

        Args:
            X (torch.Tensor): Initial condition, visible variables.
            it_mcmc (int, optional): Number of Gibbs steps to perform. If not specified it is set to 'self.gibbs_steps'. Defaults to None.
            batch_size (int, optional): Batch size. Defaults to -1.

        Returns:
            list: (visible variables, visible magnetizations, hidden variables, hidden magnetizations)
        """

        check_input_type(X)

        if not it_mcmc:
            it_mcmc = self.gibbs_steps

        v = X.clone()
        n_data = X.shape[0]
        h = torch.zeros(size=(n_data, self.Nh), device=self.device, dtype=self.dtype)
        mh = torch.zeros(size=(n_data, self.Nh), device=self.device, dtype=self.dtype)
        mv = torch.zeros(size=(n_data, self.Nv, self.num_states), device=self.device, dtype=self.dtype)
        
        if batch_size == -1:
            for _ in range(it_mcmc):
                h, mh = self.sampleHiddens(v)
                v, mv = self.sampleVisibles(h)

        else:
            num_batches = n_data // batch_size
            for m in range(num_batches):
                v_batch = v[m * batch_size : (m + 1) * batch_size]
                for _ in range(it_mcmc):
                    h_batch, mh_batch = self.sampleHiddens(v_batch)
                    v_batch, mv_batch = self.sampleVisibles(h_batch)
                v[m * batch_size : (m + 1) * batch_size] = v_batch
                h[m * batch_size : (m + 1) * batch_size] = h_batch
                mv[m * batch_size : (m + 1) * batch_size] = mv_batch
                mh[m * batch_size : (m + 1) * batch_size] = mh_batch
            # handle the remaining data
            if n_data % batch_size != 0:
                v_batch = v[num_batches * batch_size:]
                for _ in range(it_mcmc):
                    h_batch, mh_batch = self.sampleHiddens(v_batch)
                    v_batch, mv_batch = self.sampleVisibles(h_batch)
                v[num_batches * batch_size:] = v_batch
                h[num_batches * batch_size:] = h_batch
                mv[num_batches * batch_size:] = mv_batch
                mh[num_batches * batch_size:] = mh_batch

        return v, mv, h, mh
    
    def track_mc(self, X : torch.Tensor, it_mcmc=None, record_window=10, beta=1.) -> list:
        """Returns points every 'record_window' steps from the chain of lenght 'it_mcmc' starting from the initial condition X.

        Args:
            X (torch.Tensor): Initial conditions, visible variables
            it_mcmc (int, optional): Number of Gibbs steps to perform. If not specified it is set to 'self.gibbs_steps'. Defaults to None.
            record_window (int, optional): Number of steps between two consecutive records. Defaults to 10.
            beta (float, optional): Inverse temperature. Defaults to 1..

        Returns:
            list: Trajectories of (visible variables, hidden variables)
        """

        check_input_type(X)

        if not it_mcmc:
            it_mcmc = self.gibbs_steps

        history_v = []
        history_h = []
        
        v = X
        history_v.append(v.unsqueeze(0))
        
        pbar = tqdm(total=it_mcmc, colour='red')
        pbar.set_description('MCMC steps')

        h, _, = self.sampleHiddens(v, beta=beta)
        v, _ = self.sampleVisibles(h, beta=beta)
        pbar.update(1)
        history_h.append(h.unsqueeze(0))

        for t in range(it_mcmc - 1):
            
            h, _ = self.sampleHiddens(v, beta=beta)
            v, _ = self.sampleVisibles(h, beta=beta)

            if t % record_window == 0:
                history_v.append(v.unsqueeze(0))
                history_h.append(h.unsqueeze(0))

        return torch.cat(history_v, 0), torch.cat(history_h, 0)
    
    def computeDelta(self, V : torch.Tensor) -> torch.Tensor:
        """Used for computing the gradient of the Likelihood wrt the visible fields.

        Args:
            V (torch.Tensor): Visible units

        Returns:
            torch.Tensor: Frequencies of the colors at each site.
        """

        delta = (V == self.all_states).type(self.dtype).mean(1)
        delta = torch.clamp(delta, min=self.eps, max=1-self.eps) # (num_states, Nv)

        return delta
    
    def computeDeltaH(self, V : torch.Tensor, H : torch.Tensor) -> torch.Tensor:
        """Used for computing the gradient of the Likelihood wrt the weights
        of the interaction visibles-hiddens.

        Args:
            V (torch.Tensor): Visible variables.
            H (torch.Tensor): Hidden variables.

        Returns:
            torch.Tensor: Average tensor product between visible and hidden variables.
        """

        Ns = V.shape[0]
        delta = (V == self.all_states).type(self.dtype) # (num_states, Ns, Nv)
        delta = torch.clamp(delta, min=self.eps, max=1-self.eps)

        return torch.tensordot(delta, H, dims=([1],[0])) / Ns # (num_states, Nv, Nh)
    
    def computeH(self, H : torch.Tensor) -> torch.Tensor:
        """Used for computing the gradient of the Likelihood wrt the hidden fields.

        Args:
            H (torch.Tensor): Hidden variables.

        Returns:
            torch.Tensor: Average of the hidden variables.
        """

        return torch.mean(H, 0) # (1, Nh)

    def updateWeights(self, v_pos : torch.Tensor, h_pos : torch.Tensor,
                        v_neg : torch.Tensor, h_neg : torch.Tensor) -> None:
        """Computes the gradient of the Likelihood and updates the parameters.

        Args:
            v_pos (torch.Tensor): Visible variables (data).
            h_pos (torch.Tensor): Hidden variables after one Gibbs-step from the data.
            v_neg (torch.Tensor): Visible variables sampled after 'self.gibbs_steps' Gibbs-steps.
            h_neg (torch.Tensor): Hidden variables sampled after 'self.gibbs_steps' Gibbs-steps.
        """

        self.W += self.lr * (self.computeDeltaH(v_pos, h_pos) - self.computeDeltaH(v_neg, h_neg))
        self.W -= self.W.mean(0) # Gauge fixing
        self.vbias += self.lr * (self.computeDelta(v_pos) - self.computeDelta(v_neg))
        self.hbias += self.lr * (self.computeH(h_pos) - self.computeH(h_neg))

    def updateWeightsCentered(self, v_pos : torch.Tensor, h_pos : torch.Tensor,
                        v_neg : torch.Tensor, h_neg : torch.Tensor) -> None:
        """Computes the centered gradient of the Likelihood and updates the parameters.

        Args:
            v_pos (torch.Tensor): Visible variables (data).
            h_pos (torch.Tensor): Hidden variables after one Gibbs-step from the data.
            v_neg (torch.Tensor): Visible variables sampled after 'self.gibbs_steps' Gibbs-steps.
            h_neg (torch.Tensor): Hidden variables sampled after 'self.gibbs_steps' Gibbs-steps.
        """

        Ns = v_pos.shape[0]
        
        # averages over data and generated samples
        self.deltaDataAv = self.computeDelta(v_pos)
        self.hidDataAv = torch.mean(h_pos, 0)
        deltaGenAv = self.computeDelta(v_neg)
        hidGenAv = torch.mean(h_neg, 0)

        # centered variables
        delta_c_pos = (v_pos == self.all_states).type(self.dtype).transpose(0, 1) - self.deltaDataAv # (Ns, num_states, Nv)
        hid_c_pos = h_pos - self.hidDataAv # (Ns, Nh)

        delta_c_neg = (v_neg == self.all_states).type(self.dtype).transpose(0, 1) - self.deltaDataAv # (Ns, num_states, Nv)
        hid_c_neg = h_neg - self.hidDataAv # (Ns, Nh)

        # gradients
        dW = torch.tensordot(delta_c_pos, hid_c_pos, dims=[[0], [0]]) / Ns - torch.tensordot(delta_c_neg, hid_c_neg, dims=[[0], [0]]) / self.num_pcd
        dvbias = self.deltaDataAv - deltaGenAv - torch.tensordot(dW, self.hidDataAv, dims=[[2],[0]])
        dhbias = self.hidDataAv - hidGenAv - torch.tensordot(dW, self.deltaDataAv, dims=[[0, 1], [0, 1]])

        # parameters update
        self.W += self.lr * dW
        self.W -= self.W.mean(0) # Gauge fixing
        self.vbias += self.lr * dvbias
        self.hbias += self.lr * dhbias

    def iterate_mf1(self, X : torch.Tensor, alpha=1e-6, max_iter=2000, tree_mode=False, beta=1., rho=0.) -> list:
        """Iterates the mean field self-consistency equations at first order (naive mean field), starting from the visible units X, until convergence.

        Args:
            X (torch.Tensor): Initial condition (visible variables).
            alpha (float, optional): Convergence threshold. Defaults to 1e-6.
            max_iter (int, optional): Maximum number of iterations. Defaults to 2000.
            tree_mode (bool, optional): Option for the tree construction algorithm. Defaults to False.
            beta (float, optional): Inverse temperature. Defaults to 1..
            rho (float, optional): Dumping parameter. Defaults to 0..

        Returns:
            list: Fixed points of (visible magnetizations, hidden magnetizations)
        """
        
        if tree_mode:
            # In this case X is a tuple of magnetization batches
            mv, mh = X
        else:
            # In this case X is a visible units batch
            check_input_type(X)
            _, mh = self.sampleHiddens(X)
            _, mv = self.sampleVisibles(mh)

        iterations = 0

        while True:
            mv_prev = torch.clone(mv)
            mh_prev = torch.clone(mh)

            field_h = self.hbias + beta * torch.tensordot(mv, self.W, dims=[[1, 2], [1, 0]])
            mh = rho * mh_prev + (1. - rho) * torch.sigmoid(field_h)

            field_v = self.vbias + beta * torch.tensordot(mh, self.W, dims=[[1], [2]])
            mv = rho * mv_prev + (1. - rho) * torch.softmax(field_v.transpose(1, 2), 2) # (Ns, num_states, Nv)

            eps1 = torch.abs(mv - mv_prev).max()
            eps2 = torch.abs(mh - mh_prev).max()

            if max(eps1, eps2) < alpha:
                break
            iterations += 1
            if iterations >= max_iter:
                break

        return mv, mh

    def iterate_mf2(self, X : torch.Tensor, alpha=1e-6, max_iter=2000, tree_mode=False, beta=1., rho=0.) -> list:
        """Iterates the mean-field self-consistency equations at second order (TAP equations), starting from the visible units X, until convergence.

        Args:
            X (torch.Tensor): Initial condition (visible variables).
            alpha (float, optional): Convergence threshold. Defaults to 1e-6.
            max_iter (int, optional): Maximum number of iterations. Defaults to 2000.
            tree_mode (bool, optional): Option for the tree construction algorithm. Defaults to False.
            beta (float, optional): Inverse temperature. Defaults to 1..
            rho (float, optional): Dumping parameter. Defaults to 0..

        Returns:
            list: Fixed points of (visible magnetizations, hidden magnetizations)
        """

        if tree_mode:
            # In this case X is the a vector of magnetization batches
            mv, mh = X
        else:
            # In this case X is a visible units batch
            check_input_type(X)
            _, mh = self.sampleHiddens(X)
            _, mv = self.sampleVisibles(mh)
        
        W2 = torch.square(self.W)
        iterations = 0

        while True:
            mv_prev = torch.clone(mv)
            mh_prev = torch.clone(mh)

            fW = torch.einsum('svq, qvh -> svh', mv, self.W)
            
            field_h = self.hbias \
                + beta * torch.einsum('svq, qvh -> sh', mv, self.W) \
                + beta**2 * (mh - 0.5) * (
                    torch.einsum('svq, qvh -> sh', mv, W2) \
                    - torch.square(fW).sum(1))
        
            mh = rho * mh_prev + (1. - rho) * torch.sigmoid(field_h)
            Var_h = mh - torch.square(mh)
            fWm = torch.multiply(Var_h.unsqueeze(1), fW) # svh

            field_v = self.vbias \
                + beta * torch.einsum('sh, qvh -> sqv', mh, self.W) \
                + beta**2 * (0.5 * torch.einsum('sh, qvh -> sqv', Var_h, W2) \
                - torch.einsum('svh, qvh -> sqv', fWm, self.W))
            mv = rho * mv_prev + (1. - rho) * torch.softmax(field_v.transpose(1, 2), 2) # (Ns, num_states, Nv)

            eps1 = torch.abs(mv - mv_prev).max()
            eps2 = torch.abs(mh - mh_prev).max()

            if max(eps1, eps2) < alpha:
                break
                
            iterations += 1
            if iterations >= max_iter:
                break
            
        return mv, mh
    
    def iterate_mean_field(self, X : torch.Tensor, order=2, batch_size=128, alpha=1e-6, tree_mode=False, verbose=True, beta=1., rho=0., max_iter=2000) -> list:
        """Iterates the mean field self-consistency equations at the specified order, starting from the visible units X, until convergence.

        Args:
            X (torch.Tensor): Initial condition (visible variables).
            order (int, optional): Order of the expansion (1, 2). Defaults to 2.
            batch_size (int, optional): Number of samples in each batch. To set based on the memory availability. Defaults to 100.
            alpha (float, optional): Convergence threshold. Defaults to 1e-6.
            tree_mode (bool, optional): Option for the tree construction algorithm. Defaults to False.
            verbose (bool, optional): Whether to print the progress bar or not. Defaults to True.
            beta (float, optional): Inverse temperature. Defaults to 1..
            rho (float, optional): Dumping parameter. Defaults to 0..
            max_iter (int, optional): Maximum number of iterations. Defaults to 2000..

        Raises:
            NotImplementedError: If the specifiend order of expansion has not been implemented.

        Returns:
            list: Fixed points of (visible magnetizations, hidden magnetizations)
        """

        if order not in [1, 2]:
            raise NotImplementedError('Possible choices for the order parameter: (1, 2)')

        if order == 1:
            sampling_function = self.iterate_mf1
        elif order == 2:
            sampling_function = self.iterate_mf2

        if tree_mode:
            n_data = X[0].shape[0]
        else:
            check_input_type(X)
            n_data = X.shape[0]

        mv = torch.tensor([], device=self.device)
        mh = torch.tensor([], device=self.device)

        num_batches = n_data // batch_size
        num_batches_tail = num_batches
        if n_data % batch_size != 0:
            num_batches_tail += 1
            if verbose:
                pbar = tqdm(total=num_batches + 1, colour='red', ascii='-#')
                pbar.set_description('Iterating Mean Field')
        else:
            if verbose:
                pbar = tqdm(total=num_batches, colour='red', ascii='-#')
                pbar.set_description('Iterating Mean Field')
        for m in range(num_batches):
            if tree_mode:
                X_batch = []
                for mag in X:
                    X_batch.append(mag[m * batch_size : (m + 1) * batch_size, :])
            else:
                X_batch = X[m * batch_size : (m + 1) * batch_size, :]

            mv_batch, mh_batch = sampling_function(X_batch, alpha=alpha, tree_mode=tree_mode, beta=beta, rho=rho, max_iter=max_iter)
            mv = torch.cat([mv, mv_batch], 0)
            mh = torch.cat([mh, mh_batch], 0)
            
            if verbose:
                pbar.update(1)

        # handle the remaining data
        if n_data % batch_size != 0:
            if tree_mode:
                X_batch = []
                for mag in X:
                    X_batch.append(mag[num_batches * batch_size:, :])
            else:
                X_batch = X[num_batches * batch_size:, :]
                
            mv_batch, mh_batch = sampling_function(X_batch, alpha=alpha, tree_mode=tree_mode, beta=beta, rho=rho, max_iter=max_iter)
            mv = torch.cat([mv, mv_batch], 0)
            mh = torch.cat([mh, mh_batch], 0)
            
            if verbose:
                pbar.update(1)
                pbar.close()

        return mv, mh

    def compute_TAP_free_energy(self, mv : torch.Tensor, mh : torch.Tensor) -> torch.Tensor:
        """Computes the mean field free energy at the second order (TAP). mv and mh must be fixed points
        obtained from the method 'iterate_mean_field'.

        Args:
            mv (torch.Tensor): Visible magnetizations.
            mh (torch.Tensor): Hidden magnetizations.

        Returns:
            torch.Tensor: Second-order expansion of the Gibbs free energy.
        """

        Ns = mv.shape[0]

        # order zero
        zo = - torch.sum(mv * torch.log(mv), dim=(1, 2))\
            - torch.sum(mh * torch.log(mh) + (1. - mh) * torch.log(1 - mh), dim=1)

        # first order
        fo = torch.tensordot(mv, self.vbias, dims=[[1, 2], [1, 0]])\
            + torch.tensordot(mh, self.hbias, dims=[[1], [0]])\
            + torch.tensordot(mv, torch.tensordot(mh, self.W, dims=[[1], [2]]), dims=[[1, 2], [2, 1]]).diag()

        
        # second order
        W2 = torch.square(self.W)
        fW2 = torch.square(torch.multiply(mv.transpose(1, 2).unsqueeze(-1), self.W.unsqueeze(0)).sum(1)).sum(1) # (Ns, Nh)
        so = 0.5 * torch.tensordot(mv, torch.tensordot(mh - torch.square(mh), W2, dims=[[1], [2]]), dims=[[1, 2], [2, 1]]).diag()\
            - 0.5 * torch.tensordot(mh - torch.square(mh), fW2, dims=[[1], [1]]).diag()
        
        
        return (zo + fo + so) / Ns
    
    def fitBatch(self, X : torch.Tensor) -> None:
        """Updates the model's parameters using the data batch X.

        Args:
            X (torch.Tensor): Batch of data.
        """

        h_pos, _ = self.sampleHiddens(X)
        if self.CDLearning:
            # in CD mode, the Markov Chain starts from the data in the batch
            self.X_pc = X
            self.X_pc, h_neg = self.getAv()
        else:
            self.X_pc, h_neg = self.getAv()
        
        if self.updCentered:
            self.updateWeightsCentered(X, h_pos, self.X_pc, h_neg)
        else:   
            self.updateWeights(X, h_pos, self.X_pc, h_neg)
    
    def fit(self, train_dataset : torch.utils.data.Dataset, training_mode : str='PCD', epochs : int=1000, num_pcd=500, lr : float=0.001,
            batch_size : int=500, gibbs_steps : int=10, updCentered : bool=True,
            restore : bool=False) -> None:
        """Train the model.

        Args:
            train_dataset (torch.utils.data.Dataset): Training dataset.
            training_mode (str, optional): Training mode among (CD, PCD, Rdm). Defaults to 'PCD'.
            epochs (int, optional): Number of epochs for the training. Defaults to 1000.
            num_pcd (int, optional): Number of permanent chains. Defaults to 500.
            lr (float, optional): Learning rate. Defaults to 0.001.
            batch_size (int, optional): Batch size. Defaults to 500.
            gibbs_steps (int, optional): Number of MCMC steps for evaluating the gradient. Defaults to 10.
            updCentered (bool, optional): Option for centering the gradient. Defaults to True.
            restore (bool, optional): Option for restore the training of an old model. Defaults to False.
        """
        
        # initialize trainig of a new RBM
        if not restore:
            self.training_mode = training_mode
            self.ep_max = epochs
            self.lr = lr
            self.num_pcd = num_pcd
            self.mb_s = batch_size
            self.gibbs_steps = gibbs_steps
            self.updCentered = updCentered
            self.UpdByEpoch = train_dataset.__len__() // self.mb_s
            self.deltaDataAv = train_dataset.get_dataset_mean()
            self.time_start = time.time()
            
            if training_mode == 'Rdm':
                self.CDLearning = False
                self.ResetPermChainBatch = True
            elif training_mode == 'PCD':
                self.CDLearning = False
                self.ResetPermChainBatch = False
            else:
                self.CDLearning = True
                self.ResetPermChainBatch = False # this has no actual effect with CD
                
            self.X_pc = torch.randint(0, self.num_states, (self.num_pcd, self.Nv), device=self.device, dtype=torch.int64)
            self.save_RBM_state()
            
        # create dataloader
        train_dataloader = DataLoader(train_dataset, batch_size=int(self.mb_s), shuffle=True, pin_memory=True, num_workers=4, drop_last=True)

        pbar = tqdm(initial=self.ep_start, total=self.ep_max + 1, colour='red', dynamic_ncols=True, ascii='-#')
        pbar.set_description('Training RBM')
        for i in np.arange(self.ep_start, self.ep_max + 1):
            self.ep_tot += 1
            if type(lr) == list:
                self.lr = lr[i]
                
            for batch in train_dataloader:
                if self.ResetPermChainBatch:
                    self.X_pc = torch.randint(0, self.num_states, (self.num_pcd, self.Nv), device=self.device, dtype=torch.int64)

                Xb = batch.to(self.device)
                self.fitBatch(Xb)
                self.up_tot += 1

            if self.ep_tot in self.list_save_rbm:
                self.save_RBM_state()
        
        # Rename the file if updated from a previous training
        if restore:
            fname_old = self.file_stamp
            fname_new = self.generate_model_stamp(dataset_stamp=self.dataset_filename.split('/')[-1][:-3],
                                                  epochs=self.ep_max,
                                                  learning_rate=self.lr,
                                                  gibbs_steps=self.gibbs_steps,
                                                  batch_size=self.mb_s,
                                                  n_chains=self.num_pcd,
                                                  training_mode=self.training_mode)
            self.file_stamp = fname_new
            os.rename(fname_old, fname_new)
            f = h5py.File(fname_new, 'r+')
            del f['file_stamp']
            f['file_stamp'] = fname_new
            f.close()
    
    def save_RBM_state(self):
        """Saves the RBM in a .h5 file specified by 'self.file_stamp'.
        """
        
        if (len(self.list_save_rbm) > 0) & (self.ep_tot == 0):
            f = h5py.File(self.file_stamp, 'w')
            f.create_dataset('lr', data=self.lr)
            f.create_dataset('NGibbs', data=self.gibbs_steps)
            f.create_dataset('miniBatchSize', data=self.mb_s)
            f.create_dataset('numPCD', data=self.num_pcd)
            f.create_dataset('alltime', data=self.list_save_rbm)
            f.create_dataset('training_mode', data=self.training_mode)
            f.create_dataset('var_init', data=self.var_init)
            f.create_dataset('ep_max', data=self.ep_max)
            f.create_dataset('file_stamp', data=self.file_stamp)
            
            f.create_dataset('updCentered', data=self.updCentered)
            f.create_dataset('dataset_filename', data=self.dataset_filename)
            f.create_dataset('UpdByEpoch', data=self.UpdByEpoch)
            f.create_dataset('model_stamp', data=self.model_stamp)
            f.create_dataset('time_start', data=self.time_start)
            f.create_dataset('training_time', data=0)
            f.create_dataset('seed', data=self.seed)

            f['ep_tot'] = self.ep_tot
            f['up_tot'] = self.up_tot
            f['X_pc'] = self.X_pc.cpu()
            if self.updCentered:
                f['deltaDataAv'] = self.deltaDataAv.cpu()
                f['hidDataAv'] = self.hidDataAv.cpu()
            f['torch_rng_state'] = torch.get_rng_state()
            f['numpy_rng_arg0'] = np.random.get_state()[0]
            f['numpy_rng_arg1'] = np.random.get_state()[1]
            f['numpy_rng_arg2'] = np.random.get_state()[2]
            f['numpy_rng_arg3'] = np.random.get_state()[3]
            f['numpy_rng_arg4'] = np.random.get_state()[4]
            f.close()

        else:
            f = h5py.File(self.file_stamp, 'r+')
            f.create_dataset('W' + str(self.ep_tot), data=self.W.cpu())
            f.create_dataset('vbias' + str(self.ep_tot), data=self.vbias.cpu())
            f.create_dataset('hbias' + str(self.ep_tot), data=self.hbias.cpu())
            if self.training_mode == 'PCD':
                f.create_dataset('X_pc', data=self.X_pc.cpu())

            del f['ep_tot']
            del f['ep_max']
            del f['up_tot']
            del f['training_time']
            if self.updCentered:
                del f['deltaDataAv']
                del f['hidDataAv']
            del f['alltime']
            del f['torch_rng_state']
            del f['numpy_rng_arg0']
            del f['numpy_rng_arg1']
            del f['numpy_rng_arg2']
            del f['numpy_rng_arg3']
            del f['numpy_rng_arg4']
            f['ep_tot'] = self.ep_tot
            f['ep_max'] = self.ep_tot
            f['up_tot'] = self.up_tot
            if self.updCentered:
                f['deltaDataAv'] = self.deltaDataAv.cpu()
                f['hidDataAv'] = self.hidDataAv.cpu()
            f['training_time'] = time.time() - self.time_start
            self.training_time = time.time() - self.time_start
            f['alltime'] = self.list_save_rbm[self.list_save_rbm <= self.ep_tot]
            f['torch_rng_state'] = torch.get_rng_state()
            f['numpy_rng_arg0'] = np.random.get_state()[0]
            f['numpy_rng_arg1'] = np.random.get_state()[1]
            f['numpy_rng_arg2'] = np.random.get_state()[2]
            f['numpy_rng_arg3'] = np.random.get_state()[3]
            f['numpy_rng_arg4'] = np.random.get_state()[4]
            f.close()
            
    def generate_model_stamp(self, dataset_stamp : str, epochs : int, learning_rate : float,
                             gibbs_steps : int, batch_size : int, n_chains : int, training_mode : str) -> str:
        """Produces the stamp that identifies the model.

        Args:
            dataset_stamp (str): Name of the dataset.
            epochs (int): Epochs of the training.
            learning_rate (float): Learning rate.
            gibbs_steps (int): Number of MCMC steps for evaluating the gradient.
            batch_size (int): Batch size.
            training_mode (str): Training mode among (PCD, CD, Rdm).

        Returns:
            str: Model identification stamp.
        """
    
        if type(learning_rate) == list:
            lr_description = 'Adaptive'
        else:
            lr_description = learning_rate
        stamp = 'models/{0}/{1}-{2}-ep{3}-lr{4}-Nh{5}-NGibbs{6}-mbs{7}-chains{8}-{9}.h5'.format(
            dataset_stamp,
            self.model_stamp,
            dataset_stamp,
            epochs,
            lr_description,
            self.Nh,
            gibbs_steps,
            batch_size,
            n_chains,
            training_mode)
        
        return stamp
    
    def print_log(self):
        dt = datetime.datetime.strptime(self.model_stamp.split('-')[-1],'%Y.%m.%d.%H.%M')
        training_h = int(self.training_time // 3600)
        training_m = int((self.training_time % 3600) // 60)
        training_s = int(self.training_time % 60)
        message = f"""
        date: {dt.strftime('%Y/%m/%d %H:%M')}
        model: {self.model_stamp.split('-')[0]}
        dataset: {self.dataset_filename.split('/')[-1]}
        training mode: {self.training_mode}
        Nv: {self.Nv}  Nh: {self.Nh} epochs: {self.ep_tot} lr: {self.lr} gibbs_steps: {self.gibbs_steps} batch_size: {self.mb_s} num_pcd: {self.num_pcd}
        Gradient updates per epoch: {self.UpdByEpoch}
        training time: {training_h}h{training_m}m{training_s}s
        seed: {self.seed}
        """
        
        return message

def check_input_type(X : any):
    """Verifies that the input X is a Long tensor.

    Args:
        X (any): Input.

    Raises:
        TypeError: The input does not have the correct type.
    """
    if (X.dtype != torch.int64):
        raise TypeError('Input must be of type torch.int64')