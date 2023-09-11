import torch
from torch.distributions import Categorical
from torch.utils.data import DataLoader
import h5py
import numpy as np
import sys
import os
import time
from tqdm import tqdm
import datetime

class RBM:

    def __init__(self,
                num_visible=0,                  # Number or visisble variables
                num_hidden=0,                   # Number of hidden variables
                num_categ=0,                    # Number of labelled categories
                num_states=21,                   # Number of Potts states
                device= torch.device('cpu'),    # CPU or GPU?
                var_init=1e-4,                  # Variance of the initial weights
                dtype=torch.float32):           # Data type used during computations
        
        # structure-specific variables
        self.Nv = num_visible        
        self.Nh = num_hidden
        self.num_categ = num_categ
        self.device = device
        self.dtype = dtype
        self.num_states = num_states
        self.var_init = var_init
        self.seed = 0
        
        # training-specific variables
        self.gibbs_steps = 0
        self.num_pcd = 0
        self.lr = 0
        self.lr_labels = 0
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
        self.partial_labels = True
        self.lab_frac = 0.
        self.L1_reg = 0
        self.L2_reg = 0
        
        # weights of the RBM
        self.W = torch.randn(size=(self.num_states, self.Nv, self.Nh), device=self.device, dtype=self.dtype) * self.var_init
        self.D = torch.randn(size=(self.num_categ, self.Nh), device=self.device, dtype=self.dtype) * self.var_init
        # visible and hidden biases
        self.vbias = torch.zeros((self.num_states, self.Nv), device=self.device, dtype=self.dtype)
        self.hbias = torch.zeros(self.Nh, device=self.device, dtype=self.dtype)
        self.lbias = torch.zeros((self.num_categ,), device=self.device, dtype=self.dtype)
        # permanent chain
        self.X_pc = None
        self.l_pc = None
        self.deltaDataAv = torch.tensor([0])
        self.hidDataAv = torch.tensor([0])
        self.labDataAv = torch.tensor([0])
        
        # identity variables
        self.dataset_filename = ''
        timestamp = str('.'.join(list(str(time.localtime()[i]) for i in range(5))))
        self.model_stamp = f'PottsBernoulliSslRBM-{timestamp}'
        self.file_stamp = ''
        
        # auxiliary variables
        self.all_states = torch.arange(self.num_states, device=self.device).reshape(-1,1,1)
        self.all_v = torch.arange(self.Nv, device=self.device)
        self.all_categ = torch.arange(self.num_categ, device=self.device).reshape(-1, 1)
        
        # constants
        self.eps = 1e-7 # precision for clamping values
    
    def loadRBM(self, fname : str, stamp=None) -> None:
        """Loads the RBM saved in fname at t_age = stamp. If stamp is not provided, the oldest model is used.

        Args:
            fname (str): Path of the model to load.
            stamp (int, optional): Age of the RBM. If 'stamp' is not present among the saved models, the closest one is taken. Defaults to None.
        """

        f = h5py.File(fname, 'r')
        if not stamp:
            stamp = str(f['ep_tot'][()])
        else:
            stamp = str(stamp)

        # if stamp is not present, take the closest value
        alltime = f['alltime'][()]
        if int(stamp) not in alltime:
            idx = np.argmin(np.abs(alltime - int(stamp)))
            stamp = str(alltime[idx])
        self.t_age = stamp

        self.W = torch.tensor(np.array(f['W' + stamp]))
        self.D = torch.tensor(np.array(f['D' + stamp]))
        self.vbias = torch.tensor(np.array(f['vbias' + stamp]))
        self.hbias = torch.tensor(np.array(f['hbias' + stamp]))
        self.lbias = torch.tensor(np.array(f['lbias' + stamp])) 
        self.Nv = self.W.shape[1]
        self.Nh = self.W.shape[2]
        self.num_categ = self.D.shape[0]
        self.num_states = self.W.shape[0]
        self.all_v = torch.arange(self.Nv, device=self.device)
        self.all_categ = torch.arange(self.num_categ, device=self.device).reshape(-1, 1)
        self.all_states = torch.arange(self.num_states, device=self.device).reshape(-1,1,1)
        self.gibbs_steps = f['NGibbs'][()]
        self.var_init = f['var_init'][()]
        self.num_pcd = f['numPCD'][()]
        self.lr = f['lr'][()]
        self.lr_labels = f['lr_labels'][()]
        self.ep_max = f['ep_max'][()]
        self.mb_s = f['miniBatchSize'][()]
        self.ep_tot = f['ep_tot'][()]
        self.up_tot = f['up_tot'][()]
        self.L1_reg = f['L1_reg'][()]
        self.L2_reg = f['L2_reg'][()]
        self.list_save_rbm = f['alltime'][()]
        self.file_stamp = f['file_stamp'][()].decode('utf-8')
        self.updCentered = f['updCentered'][()]
        self.UpdByEpoch = f['UpdByEpoch'][()]
        self.dataset_filename = f['dataset_filename'][()].decode('utf8')
        self.model_stamp = f['model_stamp'][()].decode('utf8')
        self.partial_labels = bool(f['partial_labels'][()])
        self.lab_frac = f['lab_frac'][()]
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
            self.X_pc = torch.tensor(np.array(f['X_pc' + stamp]))
            self.l_pc = torch.tensor(np.array(f['L_pc' + stamp]))
        else:
            self.CDLearning = True
            self.ResetPermChainBatch = False # this has no actual effect with CD
        
        if self.device.type != 'cpu':
            self.W = self.W.to(self.device)
            self.D = self.D.to(self.device)
            self.vbias = self.vbias.to(self.device)
            self.hbias = self.hbias.to(self.device)
            self.lbias = self.lbias.to(self.device)
            if self.training_mode == 'PCD':
                self.X_pc = self.X_pc.to(self.device)
                self.l_pc = self.l_pc.to(self.device)
    
    def compute_energy(self, V : torch.Tensor, H : torch.Tensor, L : torch.Tensor) -> torch.Tensor:
        """Computes the Hamiltonian on the visible (V), hidden (H), and label (L) variables.

        Args:
            V (torch.Tensor): Visible variables.
            H (torch.Tensor): Hidden Variables.
            L (torch.Tensor): Labels.

        Returns:
            torch.Tensor: Energy of the data points.
        """

        check_input_type(V)
        check_input_type(L)

        fields = self.vbias[V, self.all_v].sum(1) + torch.tensordot(self.hbias, H, dims=[[0], [1]]) + self.lbias[L]
        interaction_vh = torch.bmm(self.W[V, self.all_v, :], H.unsqueeze(-1)).squeeze(-1).sum(1)
        interaction_lh = torch.multiply(H, self.D[L, :]).sum(1)

        return - fields - interaction_vh - interaction_lh
    
    def compute_energy_visibles(self, V : torch.Tensor, L : torch.Tensor) -> torch.Tensor:
        """Computes the unnormalized Log-Likelihood of the single data points.

        Args:
            V (torch.Tensor): Visible units.
            L (torch.Tensor): Labels associated to the data.

        Returns:
            torch.Tensor: Unnormalized Log-Likelihood.
        """
        
        fields = self.vbias[V, self.all_v].sum(1) + self.lbias[L]
        I_ml = self.hbias.reshape(1, self.Nh) + self.W[V, self.all_v, :].sum(1) + self.D[L] # (Ns, Nh)
        log_term = I_ml.clone()
        idx = log_term < 10
        log_term[idx] = torch.log(1. + torch.exp(I_ml[idx]))
        E = fields + log_term.sum(1) # (Ns,)
        
        return -E.type(torch.float32)
    
    def compute_partition_function_AIS(self, V : torch.tensor, n_beta : int, n_chains : int) -> float:
        """Estimates the partition function of the model using Annealed Importance Sampling.

        Args:
            V (torch.Tensor): Input data to estimate the fields of the independent-site model.
            n_beta (int): Number of inverse temperatures that define the trajectories.
            n_chains (int): Number of chains to run in parallel.

        Returns:
            float: Estimate of the partition function.
        """
        
        # define the energy of the prior pdf over the visible units (independent-site model)
        def compute_prior_features(V : torch.Tensor) -> list:
            """Returns the visible fields, energy and log partition function of an independent-site model over the visible variables.

            Args:
                V (torch.Tensor): Visible data.

            Returns:
                list: prior_biases, prior_energy, prior logZ
            """
            eps = 1e-7
            freq = (V == self.all_states).type(self.dtype).mean(1)
            freq = torch.clamp(freq, min=eps, max=1. - eps)
            vbias0 = torch.log(freq) - 1/self.num_states * torch.sum(torch.log(freq), 0)
            energy0 = vbias0[V, self.all_v].sum(1).mean()
            logZ0 = torch.log(torch.sum(torch.exp(vbias0), 0)).sum()

            return vbias0, energy0, logZ0
        
        E = torch.zeros(n_chains, device=self.device, dtype=self.dtype)
        beta_list = np.linspace(0, 1., n_beta)
        dB = 1. / n_beta
        # initialize the chains
        vbias0, energy0, logZ0_visibles = compute_prior_features(V)
        logZ0 = logZ0_visibles + self.Nh * np.log(2.) + np.log(self.num_categ) # hidden units and labels are assumed uniformly distributed
        v = Categorical(torch.softmax(vbias0.mT.unsqueeze(0).repeat(n_chains, 1, 1), -1)).sample()
        h = torch.randint(0, 2, (n_chains, self.Nh), device=self.device).type(self.dtype)
        l = torch.randint(0, self.num_categ, (n_chains,), device=self.device).type(self.dtype)
        energy1 = self.compute_energy(v, h, l)
        E += energy1 - energy0
        for beta in beta_list:
            v, _ = self.sampleVisibles(h, beta=beta)
            l, _ = self.sampleLabels(h, beta=beta)
            h, _ = self.sampleHiddens(v, l, beta=beta)
            E += self.compute_energy(v, h, l)
            
        # Subtract the average for avoiding overflow
        W = -dB * E
        W_ave = W.mean()
        logZ = logZ0 + torch.log(torch.mean(torch.exp(W - W_ave))) + W_ave
        
        return logZ
        
    def compute_log_likelihood_AIS(self, V : torch.Tensor, L : torch.Tensor, n_beta=100, n_chains=500) -> torch.Tensor:
        """Computes the unnormalized Log-Likelihood of the single data points.

        Args:
            V (torch.Tensor): Visible units.
            L (torch.Tensor): Labels associated to the data.
            n_beta (int): Number of inverse temperatures that define the trajectories.
            n_chains (int): Number of chains to run in parallel.

        Returns:
            torch.Tensor: Unnormalized Log-Likelihood.
        """
        
        check_input_type(V)
        check_input_type(L)
        logZ = self.compute_partition_function_AIS(V, n_beta=n_beta, n_chains=n_chains)
        LL = - self.compute_energy_visibles(V, L).mean() - logZ # (Ns,)
        
        return LL

    def sampleHiddens(self, V : torch.Tensor, L : torch.Tensor, beta=1.) -> list:
        """Samples the hidden variables by performing one block Gibbs sampling step.

        Args:
            V (torch.Tensor): Visible units.
            L (torch.Tensor): Labels.
            beta (float, optional): Inverse temperature. Defaults to 1.

        Returns:
            list: (hidden variables, hidden magnetizations)
        """

        # V is a batch of size (Ns, Nv)        
        I = self.W[V, self.all_v, :].sum(1) + self.D[L, :] # (Ns, Nh)
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
    
    def sampleLabels(self, H : torch.Tensor, beta=1.) -> list:
        """Samples the label variables by performing one block Gibbs sampling step.

        Args:
            H (torch.Tensor): Hidden variables.
            beta (float, optional): Inverse temperature. Defaults to 1..

        Returns:
            list: (label varaibles, label magnetizations)
        """

        z = self.lbias + torch.tensordot(self.D, H, dims=[[1], [1]]).t() # (Ns, num_categ)
        ml = torch.softmax(beta * z, 1) # (Ns, num_categ)
        l = Categorical(logits=(beta * z)).sample() # (Ns,)

        return l, ml 
    
    def sampleLabelsData(self, V : torch.Tensor, beta=1.) -> torch.Tensor:
        """Monte Carlo sampling of labels starting from data.

        Args:
            V (torch.Tensor): Visible variables.
            beta (float, optional): Inverse temperature. Defaults to 1..

        Returns:
            torch.Tensor: Sampled labels.
        """

        l = torch.randint(0, self.num_categ, (self.mb_s,))
        visible_field = self.W[V, self.all_v, :].sum(1)

        for _ in range(self.gibbs_steps):
            # sample from p(h|l; v_data)
            I = visible_field + self.D[l, :] # (Ns, Nh)
            mh = torch.sigmoid(beta * (self.hbias + I))
            h = torch.bernoulli(mh)
            # sample from p(l|h)
            z = self.lbias + torch.tensordot(self.D, h, dims=[[1], [1]]).t() # (Ns, num_categ)
            l = Categorical(logits=(beta * z)).sample()

        return l
    
    def getAv_PCD(self) -> list:
        """Performs it_mcmc Gibbs steps starting from the permanent chains. Used for the training in PCD and CD mode.

        Returns:
            list: (visible variables, visible magnetizations, hidden variables, hidden magnetizations, label variables, label magnetizations)
        """

        v = self.X_pc
        l = self.l_pc
        _, mh = self.sampleHiddens(v, l)
        v, _ = self.sampleVisibles(mh)
        l, _ = self.sampleLabels(mh)

        for _ in range(1, self.gibbs_steps):
            _, mh = self.sampleHiddens(v, l)
            v, _ = self.sampleVisibles(mh)
            l, _ = self.sampleLabels(mh)
        
        return v, mh, l
    
    def getAv_fixedLabel(self, L : torch.Tensor) -> list:
        """Performs it_mcmc Gibbs steps conditioned to the labels. Used for the training in Rdm mode.

        Args:
            L (torch.Tensor): Fixed labels.

        Returns:
            list: (visible variables, visible magnetizations, hidden variables, hidden magnetizations, label variables, label magnetizations)
        """

        v = self.X_pc
        _, mh = self.sampleHiddens(v, L)
        v, _ = self.sampleVisibles(mh)

        for _ in range(1, self.gibbs_steps):
            _, mh = self.sampleHiddens(v, L)
            v, _ = self.sampleVisibles(mh)
        
        return v, mh
    
    def getAv_fixedVisible(self, V) -> list:
        """Performs it_mcmc Gibbs steps conditioned to the visibles. Used for the training in Rdm mode.

        Args:
            V (torch.Tensor): Fixed visible variables.

        Returns:
            list: (visible variables, visible magnetizations, hidden variables, hidden magnetizations, label variables, label magnetizations)
        """

        l = self.l_pc
        _, mh = self.sampleHiddens(V, l)
        l, _ = self.sampleLabels(mh)

        for _ in range(1, self.gibbs_steps):
            _, mh = self.sampleHiddens(V, l)
            l, _ = self.sampleLabels(mh)
        
        return mh, l

    def conditioned_sampling(self, X : torch.Tensor, targets : torch.Tensor, it_mcmc : int=None, batch_size : int=-1) -> list:
        """Samples variables and magnetizatons starting from the initial condition X

        Args:
            X (torch.Tensor): Initial condition, visible variables.
            targets (torch.Tensor): Target labels.
            it_mcmc (int, optional): Number of Gibbs steps to perform. If not specified it is set to 'self.gibbs_steps'. Defaults to None.
            batch_size (int, optional): Batch size. Defaults to -1.

        Returns:
            list: (visible variables, visible magnetizations, hidden variables, hidden magnetizations, label variables, label magnetizations)
        """

        if not it_mcmc:
            it_mcmc = self.gibbs_steps
        
        v = X.clone()
        n_data = X.shape[0]
        h = torch.zeros(size=(n_data, self.Nh), device=self.device, dtype=self.dtype)
        mh = torch.zeros(size=(n_data, self.Nh), device=self.device, dtype=self.dtype)
        mv = torch.zeros(size=(n_data, self.Nv, self.num_states), device=self.device, dtype=self.dtype)
        
        if batch_size == -1:
            for _ in range(it_mcmc):
                h, mh = self.sampleHiddens(v, targets)
                v, mv = self.sampleVisibles(mh)
        
        else:
            num_batches = n_data // batch_size
            for m in range(num_batches):
                targets_batch = targets[m * batch_size : (m + 1) * batch_size]
                v_batch = v[m * batch_size : (m + 1) * batch_size]
                for _ in range(it_mcmc):
                    h_batch, mh_batch = self.sampleHiddens(v_batch, targets_batch)
                    v_batch, mv_batch = self.sampleVisibles(mh_batch)
                v[m * batch_size : (m + 1) * batch_size] = v_batch
                h[m * batch_size : (m + 1) * batch_size] = h_batch
                mv[m * batch_size : (m + 1) * batch_size] = mv_batch
                mh[m * batch_size : (m + 1) * batch_size] = mh_batch
            # handle the remaining data
            if n_data % batch_size != 0:
                targets_batch = targets[num_batches * batch_size:]
                v_batch = v[num_batches * batch_size:]
                for _ in range(it_mcmc):
                    h_batch, mh_batch = self.sampleHiddens(v_batch, targets_batch)
                    v_batch, mv_batch = self.sampleVisibles(mh_batch)
                v[num_batches * batch_size:] = v_batch
                h[num_batches * batch_size:] = h_batch
                mv[num_batches * batch_size:] = mv_batch
                mh[num_batches * batch_size:] = mh_batch

        return v, mv, h, mh
    
    def predict(self, X : torch.Tensor, L_init : torch.Tensor, it_mcmc : int=10, batch_size : int=-1) -> torch.Tensor:
        """Returns the label magnetization predicted for the input X.

        Args:
            X (torch.Tensor): Visible units input.
            L_init (torch.Tensor): Labels initialization.
            it_mcmc (int, optional): Number of iterations of the Markov Chain. Defaults to 10.
            batch_size (int, optional): Batch size. Defaults to -1.
            
        Returns:
            torch.Tensor: Label magnetizations.
        """

        l = L_init.clone()
        ml = torch.zeros(size=(X.shape[0], self.num_categ), device=self.device)
        n_data = X.shape[0]
        
        if batch_size == -1:
            for _ in range(it_mcmc):
                h, _ = self.sampleHiddens(X, l)
                l, ml = self.sampleLabels(h)
        else:
            num_batches = n_data // batch_size
            for m in range(num_batches):
                l_batch = l[m * batch_size : (m + 1) * batch_size]
                X_batch = X[m * batch_size : (m + 1) * batch_size]
                for _ in range(it_mcmc):
                    h_batch, _ = self.sampleHiddens(X_batch, l_batch)
                    l_batch, ml_batch = self.sampleLabels(h_batch)
                ml[m * batch_size : (m + 1) * batch_size] = ml_batch
            # handle the remaining data
            if n_data % batch_size != 0:
                l_batch = l[num_batches * batch_size:]
                X_batch = X[num_batches * batch_size:]
                for _ in range(it_mcmc):
                    h_batch, _ = self.sampleHiddens(X_batch, l_batch)
                    l_batch, ml_batch = self.sampleLabels(h_batch)
                ml[num_batches * batch_size:] = ml_batch
                    
        return ml
    
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
    
    def computeDeltaLabel(self, L : torch.Tensor) -> torch.Tensor:
        """Used for computing the gradient of the Likelihood wrt the label fields.

        Args:
            L (torch.Tensor): Labels.

        Returns:
            torch.Tensor: Frequencies of the label categories at each site.
        """

        delta = (L == self.all_categ).type(self.dtype).mean(1)
        delta = torch.clamp(delta, min=self.eps, max=1. - self.eps) # (num_categ, Nv)
        
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
    
    def computeDeltaLabelH(self, L : torch.Tensor, H : torch.Tensor) -> torch.Tensor:
        """Used for computing the gradient of the Likelihood wrt the weights of the interaction hiddens-labels.

        Args:
            L (torch.Tensor): Labels.
            H (torch.Tensor): Hidden units.

        Returns:
            torch.Tensor: Average tensor product between labels and hidden variables.
        """

        Ns = L.shape[0]
        delta = (L == self.all_categ).type(self.dtype) # (num_categ, Ns)
        delta = torch.clamp(delta, min=self.eps, max=1. - self.eps)

        return torch.tensordot(delta, H, dims=([1],[0])) / Ns # (num_categ, Nh)
    
    def computeH(self, H : torch.Tensor) -> torch.Tensor:
        """Used for computing the gradient of the Likelihood wrt the hidden fields.

        Args:
            H (torch.Tensor): Hidden variables.

        Returns:
            torch.Tensor: Average of the hidden variables.
        """

        return torch.mean(H, 0) # (1, Nh)

    def updateWeights(self, v_pos : torch.Tensor, l_pos : torch.Tensor, h_pos : torch.Tensor,
                        v_neg : torch.Tensor, l_neg : torch.Tensor, h_neg : torch.Tensor) -> None:
        """Computes the gradient of the Likelihood and updates the parameters.

        Args:
            v_pos (torch.Tensor): Visible variables (data).
            l_pos (torch.Tensor): Label variables (data).
            h_pos (torch.Tensor): Hidden variables after one Gibbs-step from the data.
            v_neg (torch.Tensor): Visible variables sampled after 'self.gibbs_steps' Gibbs-steps.
            l_neg (torch.Tensor): Label variables sampled after 'self.gibbs_steps' Gibbs-steps.
            h_neg (torch.Tensor): Hidden variables sampled after 'self.gibbs_steps' Gibbs-steps.
        """

        self.W += self.lr * (self.computeDeltaH(v_pos, h_pos) - self.computeDeltaH(v_neg, h_neg)) - self.L2_reg * self.W - self.L1_reg * torch.sign(self.W)
        self.W -= self.W.mean(0) # Gauge fixing
        self.D += self.lr_labels * (self.computeDeltaLabelH(l_pos, h_pos) - self.computeDeltaLabelH(l_neg, h_neg)) - self.L2_reg * self.D - self.L1_reg * torch.sign(self.D)
        self.D -= self.D.mean(0) # gauge fixing
        self.vbias += self.lr * (self.computeDelta(v_pos) - self.computeDelta(v_neg))
        self.hbias += self.lr * (self.computeH(h_pos) - self.computeH(h_neg))
        self.lbias += self.lr_labels * (self.computeDeltaLabel(l_pos) - self.computeDeltaLabel(l_neg))

    def updateWeightsCentered(self, v_pos : torch.Tensor, l_pos : torch.Tensor, h_pos : torch.Tensor,
                        v_neg : torch.Tensor, l_neg : torch.Tensor, h_neg : torch.Tensor) -> None:
        """Computes the centered gradient of the Likelihood and updates the parameters.

        Args:
            v_pos (torch.Tensor): Visible variables (data).
            l_pos (torch.Tensor): Label variables (data).
            h_pos (torch.Tensor): Hidden variables after one Gibbs-step from the data.
            v_neg (torch.Tensor): Visible variables sampled after 'self.gibbs_steps' Gibbs-steps.
            l_neg (torch.Tensor): Label variables sampled after 'self.gibbs_steps' Gibbs-steps.
            h_neg (torch.Tensor): Hidden variables sampled after 'self.gibbs_steps' Gibbs-steps.
        """

        Ns = v_pos.shape[0]
        
        # averages over data and generated samples
        self.deltaDataAv = self.computeDelta(v_pos)
        self.hidDataAv = torch.mean(h_pos, 0)
        self.labDataAv = self.computeDeltaLabel(l_pos)
        deltaGenAv = self.computeDelta(v_neg)
        hidGenAv = torch.mean(h_neg, 0)
        labGenAv = self.computeDeltaLabel(l_neg)

        # centered variables
        delta_c_pos = (v_pos == self.all_states).type(self.dtype).transpose(0, 1) - self.deltaDataAv # (Ns, num_states, Nv)
        hid_c_pos = h_pos - self.hidDataAv # (Ns, Nh)
        lab_c_pos = (l_pos == self.all_categ).type(self.dtype).transpose(0, 1) - self.labDataAv # (Ns, num_categ)

        delta_c_neg = (v_neg == self.all_states).type(self.dtype).transpose(0, 1) - self.deltaDataAv # (Ns, num_states, Nv)
        hid_c_neg = h_neg - self.hidDataAv # (Ns, Nh)
        lab_c_neg = (l_neg == self.all_categ).type(self.dtype).transpose(0, 1) - self.labDataAv # (Ns, num_categ)

        # gradients
        dW = torch.tensordot(delta_c_pos, hid_c_pos, dims=[[0], [0]]) / Ns - torch.tensordot(delta_c_neg, hid_c_neg, dims=[[0], [0]]) / v_neg.shape[0] - self.L2_reg * self.W - self.L1_reg * torch.sign(self.W)
        dD = torch.tensordot(lab_c_pos, hid_c_pos, dims=[[0], [0]]) / Ns - torch.tensordot(lab_c_neg, hid_c_neg, dims=[[0], [0]]) / l_neg.shape[0] - self.L2_reg * self.D - self.L1_reg * torch.sign(self.D)
        dvbias = self.deltaDataAv - deltaGenAv - torch.tensordot(dW, self.hidDataAv, dims=[[2],[0]])
        dhbias = self.hidDataAv - hidGenAv - torch.tensordot(dW, self.deltaDataAv, dims=[[0, 1], [0, 1]])
        dlabbias = self.labDataAv - labGenAv - torch.tensordot(dD, self.hidDataAv, dims=[[1], [0]])

        # parameters update
        self.W += self.lr * dW
        self.W -= self.W.mean(0) # Gauge fixing
        self.D += self.lr_labels * dD
        self.D -= self.D.mean(0) # gauge fixing
        self.vbias += self.lr * dvbias
        self.hbias += self.lr * dhbias
        self.lbias += self.lr_labels * dlabbias

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
            list: Fixed points of (visible magnetizations, hidden magnetizations, label magnetizations)
        """
        
        if tree_mode:
            # In this case X is a tuple of magnetization batches
            mv, mh, ml = X
        else:
            # In this case X is a visible units batch
            check_input_type(X)
            ml = self.predict(X, it_mcmc=self.gibbs_steps, verbose=False)
            L = torch.argmax(ml, 1)
            _, mh = self.sampleHiddens(X, L)
            _, mv = self.sampleVisibles(mh)

        iterations = 0

        while True:
            mv_prev = torch.clone(mv)
            mh_prev = torch.clone(mh)
            ml_prev = torch.clone(ml)

            field_h = self.hbias + beta * torch.tensordot(mv, self.W, dims=[[1, 2], [1, 0]]) + torch.tensordot(ml, self.D, dims=[[1], [0]])
            mh = rho * mh_prev + (1. - rho) * torch.sigmoid(field_h)

            field_v = self.vbias + beta * torch.tensordot(mh, self.W, dims=[[1], [2]])
            mv = rho * mv_prev + (1. - rho) * torch.softmax(field_v.transpose(1, 2), 2) # (Ns, num_states, Nv)
            
            field_l = self.lbias + torch.tensordot(mh, self.D, dims=[[1], [1]])
            ml = rho * ml_prev + (1. - rho) * torch.softmax(field_l, 1) # (Ns, n_categ)

            eps1 = torch.abs(mv - mv_prev).max()
            eps2 = torch.abs(mh - mh_prev).max()
            eps3 = torch.abs(ml - ml_prev).max()

            if max([eps1, eps2, eps3]) < alpha:
                break
            iterations += 1
            if iterations >= max_iter:
                break

        return mv, mh, ml

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
            list: Fixed points of (visible magnetizations, hidden magnetizations, label magnetizations)
        """

        if tree_mode:
            # In this case X is the a vector of magnetization batches
            mv, mh, ml = X
        else:
            # In this case X is a visible units batch
            check_input_type(X)
            ml = self.predict(X, it_mcmc=self.gibbs_steps, verbose=False)
            L = torch.argmax(ml, 1)
            _, mh = self.sampleHiddens(X, L)
            _, mv = self.sampleVisibles(mh)
        
        W2 = torch.square(self.W)
        D2 = torch.square(self.D)
        iterations = 0

        while True:
            mv_prev = torch.clone(mv)
            mh_prev = torch.clone(mh)
            ml_prev = torch.clone(ml)

            fW = torch.multiply(mv.transpose(1, 2).unsqueeze(-1), self.W.unsqueeze(0)).sum(1) # (Ns, Nv, Nh)
            cD = torch.tensordot(ml, self.D, dims=[[1], [0]]) # (Ns, Nh)
            
            field_h =  self.hbias \
                + beta * (torch.tensordot(mv, self.W, dims=[[1, 2], [1, 0]]) + torch.tensordot(ml, self.D, dims=[[1], [0]])) \
                - beta**2 * (mh - 0.5) * (
                    torch.tensordot(mv, W2, dims=[[1, 2], [1, 0]]) \
                    - torch.square(fW).sum(1) \
                    + torch.tensordot(ml, D2, dims=[[1], [0]]) \
                    - torch.square(cD))
        
            mh = rho * mh_prev + (1. - rho) * torch.sigmoid(field_h)

            mh2 = torch.square(mh)
            Var_h = mh - mh2
            mfW = torch.multiply(fW, Var_h.unsqueeze(1)) # (Ns, Nv, Nh)
            mcD = torch.multiply(cD, Var_h) # (Ns, Nh)

            field_v = self.vbias \
                + beta * torch.tensordot(mh, self.W, dims=[[1], [2]]) \
                + beta**2 * (0.5 * torch.tensordot(Var_h, W2, dims=[[1], [2]]) \
                - torch.multiply(mfW.unsqueeze(1), self.W.unsqueeze(0)).sum(-1))
            mv = rho * mv_prev + (1. - rho) * torch.softmax(field_v.transpose(1, 2), 2)
            
            field_l = self.lbias \
                + beta * torch.tensordot(mh, self.D, dims=[[1], [1]]) \
                + beta**2 * (0.5 * torch.tensordot(Var_h, D2, dims=[[1], [1]]) \
                - torch.tensordot(mcD, self.D, dims=[[1], [1]]))
            ml = rho * ml_prev + (1. - rho) * torch.softmax(field_l, 1) # (Ns, n_categ)

            eps1 = torch.abs(mv - mv_prev).max()
            eps2 = torch.abs(mh - mh_prev).max()
            eps3 = torch.abs(ml - ml_prev).max()
            
            if max([eps1, eps2, eps3]) < alpha:
                break
                
            iterations += 1
            if iterations >= max_iter:
                break

        return mv, mh, ml
    
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
            list: Fixed points of (visible magnetizations, hidden magnetizations, label magnetizations)
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
        ml = torch.tensor([], device=self.device)

        num_batches = n_data // batch_size
        if n_data % batch_size != 0:
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

            mv_batch, mh_batch, ml_batch = sampling_function(X_batch, alpha=alpha, tree_mode=tree_mode, beta=beta, rho=rho, max_iter=max_iter)
            mv = torch.cat([mv, mv_batch], 0)
            mh = torch.cat([mh, mh_batch], 0)
            ml = torch.cat([ml, ml_batch], 0)
            
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
                
            mv_batch, mh_batch, ml_batch = sampling_function(X_batch, alpha=alpha, tree_mode=tree_mode, beta=beta, rho=rho, max_iter=max_iter)
            mv = torch.cat([mv, mv_batch], 0)
            mh = torch.cat([mh, mh_batch], 0)
            ml = torch.cat([ml, ml_batch], 0)
            
            if verbose:
                pbar.update(1)
                pbar.close()

        return mv, mh, ml
    
    def fitBatchRdm(self, X : torch.Tensor, L : torch.Tensor) -> None:
        """Updates the model's parameters using the data batch (X, L).

        Args:
            X (torch.Tensor): Batch of data.
            L (torch.Tensor): Batch of labels.
        """
        
        check_input_type(X)
        check_input_type(L)
        
        if self.partial_labels:
            # identify data without labels
            no_label = (L == -1)
            # sample labels from data
            L_gen = self.sampleLabelsData(X)
            # substitute the missing labels with the sampled ones
            L[no_label] = L_gen[no_label]

        h_pos, _ = self.sampleHiddens(X, L)
        if self.CDLearning:
            # in CD mode, the Markov Chain starts from the data in the batch
            self.X_pc = X
            self.l_pc = L
            self.X_pc, h_neg, self.l_pc = self.getAv()
        else:
            # gradient with label fixed
            self.X_pc, h_neg_fl = self.getAv_fixedLabel(L)
            
            # gradient with visible configuration fixed
            h_neg_fv, self.l_pc = self.getAv_fixedVisible(X)
        
        if self.updCentered:
            self.updateWeightsCentered(X, L, h_pos, self.X_pc, L, h_neg_fl)
            self.updateWeightsCentered(X, L, h_pos, X, self.l_pc, h_neg_fv)
        else:   
            self.updateWeights(X, L, h_pos, self.X_pc, L, h_neg_fl)
            self.updateWeights(X, L, h_pos, X, self.l_pc, h_neg_fv)
    
    def fitBatchPCD(self, X : torch.Tensor, L : torch.Tensor) -> None:
        """Updates the model's parameters using the data batch (X, L).

        Args:
            X (torch.Tensor): Batch of data.
            L (torch.Tensor): Batch of labels.
        """
        
        if self.partial_labels:
            # identify data without labels
            no_label = (L == -1)
            # sample labels from data
            L_gen = self.sampleLabelsData(X)
            # substitute the missing labels with the sampled ones
            L[no_label] = L_gen[no_label]

        h_pos, _ = self.sampleHiddens(X, L)
        if self.CDLearning:
            # in CD mode, the Markov Chain starts from the data in the batch
            self.X_pc = X
            self.l_pc = L
            self.X_pc, h_neg, self.l_pc = self.getAv_PCD()
        else:
            self.X_pc, h_neg, self.l_pc = self.getAv_PCD()
        
        if self.updCentered:
            self.updateWeightsCentered(X, L, h_pos, self.X_pc, self.l_pc, h_neg)
        else:   
            self.updateWeights(X, L, h_pos, self.X_pc, self.l_pc, h_neg)
    
    def fit(self, train_dataset : torch.utils.data.Dataset, training_mode : str='PCD', epochs : int=1000, num_pcd=500, lr : float=0.001, lr_labels : float=0.001,
            batch_size : int=500, gibbs_steps : int=10, L1_reg : float=0., L2_reg : float=0., updCentered : bool=True, restore : bool=False) -> None:
        """Train the model.

        Args:
            train_dataset (torch.utils.data.Dataset): Training dataset.
            training_mode (str, optional): Training mode among (CD, PCD, Rdm). Defaults to 'PCD'.
            epochs (int, optional): Number of epochs for the training. Defaults to 100.
            num_pcd (int, optional): Number of permanent chains. Defaults to 128.
            lr (float, optional): Learning rate. Defaults to 0.001.
            lr_labels (float, optional): Learning rate of the labels matrix. Defaults to 0.001.
            batch_size (int, optional): Batch size. Defaults to 256.
            gibbs_steps (int, optional): Number of MCMC steps for evaluating the gradient. Defaults to 10.
            L1_reg (float, optional): L1 regularization parameter. Defaults to 0..
            L2_reg (float, optional): L2 regularization parameter. Defaults to 0..
            updCentered (bool, optional): Option for centering the gradient. Defaults to True.
            restore (bool, optional): Option for restore the training of an old model. Defaults to False.
        """
        
        # initialize trainig of a new RBM
        if not restore:
            self.training_mode = training_mode
            self.num_categ = train_dataset.get_num_categs()
            self.ep_max = epochs
            self.lr = lr
            self.lr_labels = lr_labels
            self.num_pcd = num_pcd
            self.mb_s = batch_size
            self.gibbs_steps = gibbs_steps
            self.updCentered = updCentered
            self.L1_reg = L1_reg
            self.L2_reg = L2_reg
            self.UpdByEpoch = train_dataset.__len__() // self.mb_s # number of batches
            self.deltaDataAv = train_dataset.get_dataset_mean()
            # Track if there are missing labels in the dataset and how many
            self.partial_labels = train_dataset.partial_labels
            self.lab_frac = train_dataset.get_labels_frac()
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
            self.l_pc = torch.randint(0, self.num_categ, (self.num_pcd,), device=self.device, dtype=torch.int64)
            self.save_RBM_state()
            
        # create dataloader
        train_dataloader = DataLoader(train_dataset, batch_size=int(self.mb_s), shuffle=True, pin_memory=True, num_workers=4, drop_last=True)

        pbar = tqdm(initial=self.ep_start, total=self.ep_max + 1, colour='red', dynamic_ncols=True, ascii='-#')
        pbar.set_description('Training RBM')
        for i in np.arange(self.ep_start, self.ep_max + 1):
            pbar.update(1)
            self.ep_tot += 1
            if type(lr) == list:
                self.lr = lr[i]
                
            for batch in train_dataloader:
                if self.ResetPermChainBatch:
                    self.X_pc = torch.randint(0, self.num_states, (self.num_pcd, self.Nv), device=self.device, dtype=torch.int64)
                    self.l_pc = torch.randint(0, self.num_categ, (self.num_pcd,), device=self.device, dtype=torch.int64)

                Xb, Lb = batch
                Xb = Xb.to(self.device)
                Lb = Lb.to(self.device)
                if self.training_mode == 'Rdm':
                    self.fitBatchRdm(Xb, Lb)
                else:
                    self.fitBatchPCD(Xb, Lb)
                self.up_tot += 1

            if self.ep_tot in self.list_save_rbm:
                self.save_RBM_state()
        
        # Rename the file if updated from a previous training
        if restore:
            fname_old = self.file_stamp
            fname_new = self.generate_model_stamp(dataset_stamp=self.dataset_filename.split('/')[-1][:-3],
                                                  epochs=self.ep_max,
                                                  learning_rate=self.lr,
                                                  learning_rate_labels=self.lr_labels,
                                                  gibbs_steps=self.gibbs_steps,
                                                  batch_size=self.mb_s,
                                                  partial_labels=self.partial_labels,
                                                  perc_labels=self.lab_frac,
                                                  L1_reg=self.L1_reg,
                                                  L2_reg=self.L2_reg,
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
            f.create_dataset('lr_labels', data=self.lr_labels)
            f.create_dataset('partial_labels', data=self.partial_labels)
            f.create_dataset('lab_frac', data=self.lab_frac)
            f.create_dataset('NGibbs', data=self.gibbs_steps)
            f.create_dataset('miniBatchSize', data=self.mb_s)
            f.create_dataset('numPCD', data=self.num_pcd)
            f.create_dataset('L1_reg', data=self.L1_reg)
            f.create_dataset('L2_reg', data=self.L2_reg)
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
            if self.updCentered:
                f['deltaDataAv'] = self.deltaDataAv.cpu()
                f['hidDataAv'] = self.hidDataAv.cpu()
                f['labDataAv'] = self.labDataAv.cpu()
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
            f.create_dataset('D' + str(self.ep_tot), data=self.D.cpu())
            f.create_dataset('vbias' + str(self.ep_tot), data=self.vbias.cpu())
            f.create_dataset('hbias' + str(self.ep_tot), data=self.hbias.cpu())
            f.create_dataset('lbias' + str(self.ep_tot), data=self.lbias.cpu())
            if self.training_mode == 'PCD':
                f.create_dataset('X_pc' + str(self.ep_tot), data=self.X_pc.cpu())
                f.create_dataset('L_pc' + str(self.ep_tot), data=self.l_pc.cpu())

            del f['ep_tot']
            del f['ep_max']
            del f['up_tot']
            del f['training_time']
            if self.updCentered:
                del f['deltaDataAv']
                del f['hidDataAv']
                del f['labDataAv']
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
                f['labDataAv'] = self.labDataAv.cpu()
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
                             learning_rate_labels : float, gibbs_steps : int, batch_size : int,
                             partial_labels : bool, perc_labels : float, training_mode : str,
                             L1_reg : str, L2_reg : str) -> str:
        """Produces the stamp that identifies the model.

        Args:
            dataset_stamp (str): Name of the dataset.
            epochs (int): Epochs of the training.
            learning_rate (float): Learning rate.
            learning_rate_labels (float): Learning rate of the labels matrix.
            gibbs_steps (int): Number of MCMC steps for evaluating the gradient.
            batch_size (int): Batch size.
            partial_labels (bool): Wheather in the original dataset there were missing labels or not.
            perc_labels (float): Ratio of the labels present.
            training_mode (str): Training mode among (PCD, CD, Rdm).
            L1_reg (str): L1 regularization parameter.
            L2_reg (str): L2 regularization parameter.

        Returns:
            str: Model identification stamp.
        """
            
        if type(learning_rate) == list:
            lr_description = 'Adaptive'
        else:
            lr_description = learning_rate
        if partial_labels:
            stamp = 'models/{0}/{1}-{2}-ep{3}-lr{4}-Nh{5}-NGibbs{6}-mbs{7}-partialLabels{8}-percLabels{9}-lr_labels{10}-L1{11}-L2{12}-{13}.h5'.format(
                dataset_stamp, self.model_stamp, dataset_stamp, epochs, lr_description, self.Nh, gibbs_steps,
                batch_size, partial_labels, perc_labels, learning_rate_labels, L1_reg, L2_reg, training_mode)
        else:
            stamp = 'models/{0}/{1}-{2}-ep{3}-lr{4}-Nh{5}-NGibbs{6}-mbs{7}-partialLabels{8}-lr_labels{9}-L1{10}-L2{11}-{12}.h5'.format(
                dataset_stamp, self.model_stamp, dataset_stamp, epochs, lr_description, self.Nh, gibbs_steps,
                batch_size, partial_labels, learning_rate_labels, L1_reg, L2_reg, training_mode)
        
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
        Nv: {self.Nv}  Nh: {self.Nh} epochs: {self.ep_tot} lr: {self.lr} lr_labels: {self.lr_labels} gibbs_steps: {self.gibbs_steps} batch_size: {self.mb_s} num_pcd: {self.num_pcd} L1_reg: {self.L1_reg} L2_reg: {self.L2_reg}
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