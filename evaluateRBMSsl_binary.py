# #!/usr/bin/python3

import sys
import os
sys.path.append(os.getcwd() + '/src')
sys.path.append(os.getcwd() + '/src/RBMs')
from BernoulliBernoulliSslRBM import RBM
import torch
import fef.scores as scores
import argparse
import utilities
from h5py import File
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib import cm
from pathlib import Path
import logging
from tabulate import tabulate
from seaborn import boxplot
import gzip
from sklearn.metrics import confusion_matrix

# use LaTeX fonts in the plots
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'serif'
plt.rcParams.update({'font.size': 12})
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{bm}'

logger = logging.getLogger()
logger.setLevel(logging.INFO)

formatter = logging.Formatter(
    "%(asctime)s | %(message)s",
    datefmt="%y/%m/%d %H:%M:%S",
)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

def plot_scores(fname, rbm, label, count, t_ages, record_times, category2label,
                score_LL : scores.SCORE,
                score_spectra : scores.SCORE,
                score_AAI : scores.SCORE,
                score_entropy : scores.SCORE,
                score_first_moment : scores.SCORE,
                score_second_moment : scores.SCORE):
    
    gibbs_steps = rbm.gibbs_steps
    training_mode = rbm.training_mode
    UpdByEpoch = rbm.UpdByEpoch
    
    n_curves = len(t_ages)
    colors = cm.get_cmap('RdYlBu', n_curves)
    fig, ax = plt.subplots(6, 1, sharex=True, dpi=100, figsize=(6, 16))
    
    ax[0].set_ylabel(r'$\epsilon^{\mathrm{E}}$', size=15)
    ll = category2label[label].replace('_', '\_')
    ax[0].set_title(f'{training_mode}-{gibbs_steps}' + r' $(k=$' + f'{gibbs_steps})\nLabel: {ll}, \# data: {count}', size=15)
    ax[0].set_xscale('log')
    ax[0].set_yscale('log')
    
    ax[1].set_ylabel(r'$\epsilon^{\mathrm{s}}$', size=15)
    ax[1].set_xscale('log')
    ax[1].set_yscale('log')
    
    ax[2].set_ylabel(r'$\Delta S$', size=15)
    ax[2].axhline(y=0., ls='dashed', color='black', alpha=0.5)
    ax[2].set_xscale('log')
    
    ax[3].set_ylabel(r'$\epsilon^{\mathrm{AAI}}$', size=15)
    ax[3].set_xscale('log')
    ax[3].set_yscale('log')

    
    ax[4].set_ylabel(r'$\epsilon^{(1)}$', size=15)
    ax[4].set_xscale('log')
    ax[4].set_yscale('log')
    
    ax[5].set_ylabel(r'$\epsilon^{(2)}$', size=15)
    ax[5].ticklabel_format(axis='y', style='sci', scilimits=(1,2))
    ax[5].set_xlabel(r'$t_{\mathrm{G}}$ [MCMC steps]', size=15)
    ax[5].set_xscale('log')
    ax[5].set_yscale('log')
    
    for i, t_age in enumerate(t_ages):
        ax[0].plot(record_times, score_LL.get_label_score(t_age, label), label=r'$t_{\mathrm{age}}=$' + str(t_age * UpdByEpoch), c=colors(i))
        ax[1].plot(record_times, score_spectra.get_label_score(t_age, label), c=colors(i))
        ax[2].plot(record_times, score_entropy.get_label_score(t_age, label), c=colors(i))
        ax[3].plot(record_times, score_AAI.get_label_score(t_age, label), c=colors(i))
        ax[4].plot(record_times, score_first_moment.get_label_score(t_age, label), c=colors(i))
        ax[5].plot(record_times, score_second_moment.get_label_score(t_age, label), c=colors(i))
    
    ncol = (n_curves // 2)
    
    ax[0].legend(bbox_to_anchor=(0.5, 1.8), loc="upper center", fontsize=12, ncol=ncol)
    
    plt.subplots_adjust(right=0.95)
    fig.text(1, 0.5, '$\mathrm{' + rbm.file_stamp.split('/')[-1].replace('_', '\_') + '}$', ha='right', va='center', rotation=90, size=15)
    
    fig.savefig(str(fname) + '.svg')
    fig.savefig(str(fname) + '.png', bbox_inches='tight')
    plt.close()

def get_alltime(fname_model : str):
    """Returns the ages of the models that are saved in `fname_model`. This bypasses the `alltime` object, and it therefore works even if the 
    training of the model was interrupted.

    Args:
        fname_model (str): Filename of the RBM model.

    Returns:
        alltime (np.ndarray): Ordered set of ages saved in the file.
    """
    
    f = File(fname_model, 'r')
    alltime = [int(s.replace('W', '')) for s in f.keys() if ('W' in s and '_prev' not in s)]
    
    return np.sort(alltime)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Generates the scores for a specified RBM model.')
    parser._action_groups.pop()
    required = parser.add_argument_group('Required arguments')
    optional = parser.add_argument_group('Optional arguments')
    
    required.add_argument('-m', '--model', type=Path, help='Path to RBM model.', required=True)
    required.add_argument('-o', '--folder', type=Path, help='Path to output folder.', required=True)
    required.add_argument('-d', '--data', type=Path, help='Path to data file.', required=True)
    required.add_argument('-t', '--gen_time', type=int, default=100000, help='(Defaults to 100000). Maximum generation time.', required=True)
    
    optional.add_argument('--data_set', type=str, default='test', help='(Defaults to `test`). Weather to use the training set or the test set for the evaluation.', choices=['train', 'test'])
    optional.add_argument('--num_samples', type=int, default=-1, help='(Defaults to -1). Specifies the number of samples to use for the evaluation. Use `-1` for the whole training set.')
    optional.add_argument('--num_points_trajectory', type=int, default=50, help='(Defaults to 50). Number of scores evaluations along the trining history.')
    optional.add_argument('--batch_size', type=int, default=300, help='(Defaults to 300). Batch size used for generating the data and for evaluating the LL score. To be set based on the memory constraints.')
    optional.add_argument('--replication_factor', type=int, default=2, help='(Defaults to 2). Multiplicative factor on the number of samples to generate wrt the dataset. Used for generating diverse samples of those of the dataset.')
    args = parser.parse_args()
    
    # device setup
    device = utilities.select_device()
    
    # import data and RBM model
    if not args.model.exists():
        raise FileNotFoundError(args.model)
    if not args.data.exists():
        raise FileNotFoundError(args.data)
    output = args.folder.joinpath(args.model.stem)
    scores_dir = output / 'scores'
    output.mkdir(exist_ok=True)
    scores_dir.mkdir(exist_ok=True)
    
    f_data = File(args.data, 'r')
    data_type = torch.float32
    
    # specify the ages of the model to use
    alltime = get_alltime(args.model)
    len_divisors = np.array([n  for n in range(1, len(alltime) // 2 + 1) if len(alltime) % n == 0])
    len_divisors = np.append(len_divisors, len(alltime))
    ncols = len_divisors[np.argmin(np.abs(len_divisors - 15))]
    print('\nAges of the saved models:')
    print(tabulate(alltime.reshape(-1, ncols)))
    t_ages = list(map(int, input('\nInsert the list of ages you want to study (separated by commas): ').split(',')))
    # checking that the ages are present among the saved models
    for t_age in t_ages:
        if t_age not in alltime:
            raise KeyError(t_age)
    
    logger.info('Loading data and RBM model')
    
    # Import the dataset
    allD = torch.tensor(f_data[args.data_set][()]).type(data_type)
    allLabels_string = f_data[args.data_set + '_labels'].asstr()[()]
    if args.num_samples != -1:
        idxs = np.random.choice(np.arange(allD.shape[0]), args.num_samples, replace=False)
        allD = allD[idxs].to(device)
        allLabels_string = allLabels_string[idxs]
    else:
        allD = allD.to(device)
    
    # Take only the data that have a label
    filtered_idxs = np.where((allLabels_string != '-1'))[0]
    allLabels_string_filtered = allLabels_string[filtered_idxs]
    label2category = {lab : i for i, lab in enumerate(np.unique(allLabels_string_filtered))}
    category2label = {i : lab for i, lab in enumerate(np.unique(allLabels_string_filtered))}
    labels_data = np.array(([label2category[lab] for lab in allLabels_string_filtered]))
    labels_data2counts = {l : c for l, c in zip(*np.unique(labels_data, return_counts=True))}
    # Take the labels present in the dataset and repeat them replication_factor times to produce the targets
    targets_oversampled = torch.tensor(labels_data, device=device, dtype=torch.int64).repeat(args.replication_factor)
    targets_prediction = torch.tensor(labels_data, device=device) # used for computing the accuracy of the predictions
    D = allD[filtered_idxs]
    ndata = D.shape[0]
    if args.batch_size > ndata:
        args.batch_size = -1
        
    unique_labels, label_counts = torch.unique(targets_prediction, return_counts=True)
    unique_labels = unique_labels.cpu().numpy()
    n_categories = unique_labels.shape[0]
    f_data.close()
    
    # import the RBM model
    rbm = RBM(num_visible=0, num_hidden=0, device=device)
    
    logger.info(f'Extracted {ndata} labelled samples from the dataset.')
    logger.info('Computing the observables on the dataset')
        
    # compute observables on the dataset
    entropy_D_label = {category2label[l] : 0 for l in unique_labels}
    for label in unique_labels:
        mask = np.where(labels_data == label)[0]
        entropy_D_label[category2label[label]] = len(gzip.compress(D[mask].int().cpu().numpy())) / len(mask)
        
    D_lengths = torch.sum(D != 0, dim=1).cpu().numpy()
    D_lengths_mean = D_lengths.mean()
    D_lengths_std = D_lengths.std()
    
    # Define the generation times at which evaluating the model
    exponent = int(np.log10(args.gen_time))
    record_times = np.unique(np.logspace(0, exponent, args.num_points_trajectory).astype(np.int64))
    record_times = np.unique(np.sort(np.append([0, rbm.gibbs_steps], record_times)))
    # For tracking all the generation process
    if args.num_points_trajectory == args.gen_time:
        record_times = np.arange(0, args.num_points_trajectory + 1)
    
    # defining scores
    score_LL = scores.SCORE(t_ages=t_ages, record_times=record_times, labels=unique_labels)
    score_spectra = scores.SCORE(t_ages=t_ages, record_times=record_times, labels=unique_labels)
    score_entropy = scores.SCORE(t_ages=t_ages, record_times=record_times, labels=unique_labels)
    score_AAI_data = scores.SCORE(t_ages=t_ages, record_times=record_times, labels=unique_labels)
    score_AAI_gen = scores.SCORE(t_ages=t_ages, record_times=record_times, labels=unique_labels)
    score_AAI = scores.SCORE(t_ages=t_ages, record_times=record_times, labels=unique_labels)
    score_first_moment = scores.SCORE(t_ages=t_ages, record_times=record_times, labels=unique_labels)
    score_second_moment = scores.SCORE(t_ages=t_ages, record_times=record_times, labels=unique_labels)
    
    accuracies = []
    predicted_labels = []
    G_lengths_mean = []
    G_lengths_std = []

    t_start = time.time()
    
    # Compute scores at different epochs
    for t_age in t_ages:

        accuracy = []
        G_lengths_mean_traj = []
        G_lengths_std_traj= []
        
        rbm.loadRBM(args.model, stamp=t_age)
        
        G_oversampled = torch.randint(0, rbm.num_states, size=(targets_oversampled.shape[0], D.shape[1]), device=device, dtype=data_type)
        predictions = torch.randint(0, rbm.num_categ, (D.shape[0],), device=device, dtype=torch.int64)
        logger.info(f'Evaluating age {t_age}')
        #pbar = tqdm(desc='Analyzing generation', total=len(record_times) - 1, leave=False, colour='red', dynamic_ncols=True, ascii='-#', file=sys.stdout)
        for tg_index in range(1, len(record_times)):
            print(f'Analyzing generation time {tg_index} of {len(record_times)}', end='\r')
            dt = (record_times[tg_index] - record_times[tg_index - 1])
            G_oversampled, _, _, _ = rbm.conditioned_sampling(G_oversampled, targets_oversampled, it_mcmc=dt, batch_size=args.batch_size)
            predictions = rbm.predict(D, predictions, it_mcmc=dt, batch_size=args.batch_size).argmax(-1)
            
            # remove duplicates
            G_oversampled_unique, unique_idxs = np.unique(G_oversampled.cpu().numpy(), axis=0, return_index=True)
            G_oversampled_unique = torch.from_numpy(G_oversampled_unique).to(device) # Caution: as it is, this tensor is sorted, i.e., biased
            targets_oversampled_unique = targets_oversampled[unique_idxs].cpu().numpy()
            
            # extract from the generated set, for each label, the same number of samples present in the datset
            G = []
            gen_labels = []
            gen_labels2count = {l : 0 for l in unique_labels}
            for g, l in zip(G_oversampled_unique, targets_oversampled_unique):
                if not torch.any(torch.all(g == D, dim=1)):
                    if gen_labels2count[l] < labels_data2counts[l]:
                        G.append(g.unsqueeze(0))
                        gen_labels.append(l)
                        gen_labels2count[l] += 1
            
            G = torch.cat(G, 0).to(device)
            gen_labels = np.array(gen_labels)
            
            for label in unique_labels:
                idxs_G = np.where(gen_labels == label)[0]
                idxs_D = np.where(labels_data == label)[0]
                
                if len(idxs_G) >= len(idxs_D):
                    random_choice = np.random.choice(np.arange(len(idxs_G)), len(idxs_D), replace=False)
                    idxs_G = idxs_G[random_choice]
                else:
                    #sys.stderr.flush()
                    #tqdm.write(f'Warning: label {category2label[label]} has {len(idxs_G)} samples in generated and {len(idxs_D)} samples in data. You may want to increase the replication_factor.', file=sys.stdout)
                    random_choice = np.random.choice(np.arange(len(idxs_D)), len(idxs_G), replace=False)
                    idxs_D = idxs_D[random_choice]
                
                LL = scores.LL_score_batched_Ssl(D[idxs_D], labels_data[idxs_D], G[idxs_G], gen_labels[idxs_G], rbm, args.batch_size).cpu().numpy()
                score_LL.update(t_age=t_age, label=label, value=LL)
                
                spectra = scores.spectrum_score(D[idxs_D], G[idxs_G]).cpu().numpy()
                score_spectra.update(t_age=t_age, label=label, value=spectra)
                
                AAI_data, AAI_gen = scores.AAI_score(D[idxs_D], G[idxs_G])
                AAI = (AAI_data - 0.5)**2 + (AAI_gen - 0.5)**2
                score_AAI.update(t_age=t_age, label=label, value=AAI.cpu().numpy())
                score_AAI_data.update(t_age=t_age, label=label, value=AAI_data.cpu().numpy())
                score_AAI_gen.update(t_age=t_age, label=label, value=AAI_gen.cpu().numpy())
                
                entropy = scores.entropy_score(entropy_D_label[category2label[label]], G[idxs_G])
                score_entropy.update(t_age=t_age, label=label, value=entropy)
                
                first_moment = scores.first_moment_score(D[idxs_D], G[idxs_G]).cpu().numpy()
                score_first_moment.update(t_age=t_age, label=label, value=first_moment)
                
                second_moment = scores.second_moment_score(D[idxs_D], G[idxs_G]).cpu().numpy()
                score_second_moment.update(t_age=t_age, label=label, value=second_moment)
    
            accuracy.append(torch.sum(predictions == targets_prediction) / ndata)
            
            G_lengths = torch.sum(G != 0, dim=1).cpu().numpy()
            G_lengths_mean_traj.append(G_lengths.mean())
            G_lengths_std_traj.append(G_lengths.std())
            
            sys.stdout.flush()
            
        accuracies.append(torch.tensor(accuracy).unsqueeze(0))
        predicted_labels.append(predictions.cpu())
        
        G_lengths_mean.append(G_lengths_mean_traj)
        G_lengths_std.append(G_lengths_std_traj)
        
    accuracies = torch.cat(accuracies, 0)
    record_times = record_times[1:]

    # plot label-specific scores
    for label, label_count in zip(unique_labels, label_counts):
        fname = output / f'score_curves-{category2label[label]}'
        plot_scores(fname, rbm, label, label_count, t_ages, record_times, category2label,
                score_LL,
                score_spectra,
                score_AAI,
                score_entropy,
                score_first_moment,
                score_second_moment)
    
    ###################################################################
    
    # Compute copy rates
    def get_copy_rates(rbm, train_data, n_tests, n_gen_categ):
        inter_copy = [] # ratios of unique generated data that are also present into the training set
        intra_copy = [] # ratios of non-unique generated data
        
        targets = []
        for l in range(rbm.num_categ):
            targets.append(torch.full((n_gen_categ,), l))
        targets = torch.cat(targets, 0).type(torch.int64).to(device)
    
        for _ in range(n_tests):
            rand_init = torch.randint(0, rbm.num_states, size=(len(targets), D.shape[1]), device=device, dtype=data_type)
            gen_data, _, _, _ = rbm.conditioned_sampling(rand_init, targets, batch_size=args.batch_size)

            n_gen_old = gen_data.shape[0]
            gen_data = torch.unique(gen_data, dim=0)
            n_gen_new = gen_data.shape[0]

            n_hits = 0
            for gen_sample in gen_data:
                if torch.any(torch.all(train_data == gen_sample, dim=1)):
                    n_hits += 1

            inter_copy.append(round(n_hits / n_gen_new, 2))
            intra_copy.append((n_gen_old - n_gen_new) / n_gen_old)

        return np.array(inter_copy), np.array(intra_copy)
    
    all_train_data = torch.tensor(File(args.data, 'r')['train'][()], dtype=torch.int64).to(device)
    
    n_tests = 50 # number of independent generations for collecting the statistics
    intra_copy_distributions = []
    inter_copy_distributions = []

    logger.info('Computing copy rates')
    #pbar = tqdm(total=len(t_ages), desc='Evaluating models', colour='green', dynamic_ncols=True, ascii='-#', leave=True, file=sys.stdout)
    for i, age in enumerate(t_ages):
        print(f'Evaluating models {i+1} of {len(t_ages)}', end='\r')
        rbm.loadRBM(args.model, stamp=age)
        inter_rates, intra_rates = get_copy_rates(rbm, all_train_data, n_tests, n_gen_categ=100)
        inter_copy_distributions.append(inter_rates)
        intra_copy_distributions.append(intra_rates)
        sys.stdout.flush()
    
    ####################################################################
    
    # Eigenvalue profiles
    
    logger.info('Computing eigenvalue profiles')
    f_model = File(args.model, 'r')
    alls_W = []
    alls_D = []
    for ep in alltime:       
        W = torch.tensor(f_model['W' + str(ep)][()])   
        D = torch.tensor(f_model['D' + str(ep)][()])
        s_W = torch.linalg.svdvals(W) 
        s_D = torch.linalg.svdvals(D) 
        alls_W.append(s_W.unsqueeze(-1))
        alls_D.append(s_D.unsqueeze(-1))
    alls_W = torch.cat(tuple(alls_W), dim=1)
    alls_D = torch.cat(tuple(alls_D), dim=1)
    f_model.close()
        
    ####################################################################

    # Plot the results
    logger.info('Plotting the results')
    
    # Generate the plots with the scores
    gibbs_steps = rbm.gibbs_steps
    training_mode = rbm.training_mode
    UpdByEpoch = rbm.UpdByEpoch
    
    n_curves = len(t_ages)
    colors = cm.get_cmap('RdYlBu', n_curves)
    fig, ax = plt.subplots(6, 1, sharex=True, dpi=100, figsize=(6, 16))
    
    ax[0].set_ylabel(r'$\epsilon^{\mathrm{E}}$', size=15)
    ax[0].set_title(f'{training_mode}-{gibbs_steps}' + r' $(k=$' + f'{gibbs_steps})', size=15)
    ax[0].set_xscale('log')
    ax[0].set_yscale('log')
    
    ax[1].set_ylabel(r'$\epsilon^{\mathrm{S}}$', size=15)
    ax[1].set_xscale('log')
    ax[1].set_yscale('log')
    
    ax[2].set_ylabel(r'$\Delta S$', size=15)
    ax[2].axhline(y=0., ls='dashed', color='black', alpha=0.5)
    ax[2].set_xscale('log')
    
    ax[3].set_ylabel(r'$\epsilon^{\mathrm{AAI}}$', size=15)
    ax[3].set_xscale('log')
    ax[3].set_yscale('log')
    
    ax[4].set_ylabel(r'$\epsilon^{(1)}$', size=15)
    ax[4].set_xscale('log')
    ax[4].set_yscale('log')
    
    ax[5].set_ylabel(r'$\epsilon^{(2)}$', size=15)
    ax[5].ticklabel_format(axis='y', style='sci', scilimits=(1,2))
    ax[5].set_xlabel(r'$t_{\mathrm{G}}$ [MCMC steps]', size=15)
    ax[5].set_xscale('log')
    ax[5].set_yscale('log')
    
    for i, t_age in enumerate(t_ages):
        ax[0].plot(record_times, score_LL.get_mean_score(t_age), label=r'$t_{\mathrm{age}}=$' + str(t_age * UpdByEpoch), c=colors(i))
        ax[1].plot(record_times, score_spectra.get_mean_score(t_age), c=colors(i))
        ax[2].plot(record_times, score_entropy.get_mean_score(t_age), c=colors(i))
        ax[3].plot(record_times, score_AAI.get_mean_score(t_age), c=colors(i))
        ax[4].plot(record_times, score_first_moment.get_mean_score(t_age), c=colors(i))
        ax[5].plot(record_times, score_second_moment.get_mean_score(t_age), c=colors(i))
    
    ncol = (n_curves // 2)
    
    ax[0].legend(bbox_to_anchor=(0.5, 1.8), loc="upper center", fontsize=12, ncol=ncol)
    
    plt.subplots_adjust(right=0.95)
    fig.text(1, 0.5, '$\mathrm{' + rbm.file_stamp.split('/')[-1].replace('_', '\_') + '}$', ha='right', va='center', rotation=90, size=15)
    
    fig.savefig(output / 'score_curves.svg')
    fig.savefig(output / 'score_curves.png', bbox_inches='tight')
    plt.close()
    
    ##########################################################################
    
    # Plot AAI score
    
    fig, ax = plt.subplots(dpi=100, figsize=(15,6), nrows=1, ncols=2)
    ax[0].set_xscale('log')
    ax[0].set_xlabel(r'$t_{\mathrm{G}}$ [MCMC steps]', size=15)
    ax[0].set_ylabel('AAI data', size=20)
    ax[0].axhline(y=0.5, ls='dashed', c='black')
    for i, t_age in enumerate(t_ages):
        ax[0].plot(record_times, score_AAI_data.get_mean_score(t_age), c=colors(i), lw=3, label=r'$t_{\mathrm{age}}=$' + str(t_age * UpdByEpoch))
    lines, labels = ax[0].get_legend_handles_labels()
        
    ax[1].set_xscale('log')
    ax[1].set_xlabel(r'$t_{\mathrm{G}}$ [MCMC steps]', size=15)
    ax[1].set_ylabel('AAI generated', size=20)
    ax[1].axhline(y=0.5, ls='dashed', c='black')
    for i, t_age in enumerate(t_ages):
        ax[1].plot(record_times, score_AAI_gen.get_mean_score(t_age), c=colors(i), lw=3, label=r'$t_{\mathrm{age}}=$' + str(t_age * UpdByEpoch))
        
    ncol = (n_curves + 1) // 2
    fig.legend(lines, labels, bbox_to_anchor=(0.5, 1.), loc="upper center", fontsize=12, ncol=ncol)
    plt.subplots_adjust(right=0.95)
    fig.text(1, 0.5, '$\mathrm{' + rbm.file_stamp.split('/')[-1].replace('_', '\_') + '}$', ha='right', va='center', rotation=90, size=5)
    fig.savefig(output / 'AAI_curves.svg')
    fig.savefig(output / 'AAI_curves.png', bbox_inches='tight')
    plt.close()
    
    ##########################################################################
    
    # Plot lengths distributions
    
    fig, ax = plt.subplots(dpi=100, figsize=(15,6), nrows=1, ncols=2)
    lines_labels = []
    ax[0].set_xscale('log')
    ax[0].set_xlabel(r'$t_{\mathrm{G}}$ [MCMC steps]', size=15)
    ax[0].set_ylabel('length mean', size=20)
    ax[0].axhline(y=D_lengths_mean, ls='dashed', c='black', label='Data')
    
    ax[1].set_xscale('log')
    ax[1].set_xlabel(r'$t_{\mathrm{G}}$ [MCMC steps]', size=15)
    ax[1].set_ylabel('length std', size=20)
    ax[1].axhline(y=D_lengths_std, ls='dashed', c='black')
    
    for i, t_age in enumerate(t_ages):
        ax[0].plot(record_times, G_lengths_mean[i], c=colors(i), lw=3, label=r'$t_{\mathrm{age}}=$' + str(t_age * UpdByEpoch))
        ax[1].plot(record_times, G_lengths_std[i], c=colors(i), lw=3)
    lines, labels = ax[0].get_legend_handles_labels()
        
    ncol = (n_curves + 1) // 2
    fig.legend(lines, labels, loc="upper center", fontsize=12, ncol=ncol)
    plt.subplots_adjust(right=0.95)
    fig.text(1, 0.5, '$\mathrm{' + args.model.name.replace('_', '\_') + '}$', ha='right', va='center', rotation=90, size=5)
    fig.savefig(output / 'length_curves.svg')
    fig.savefig(output / 'length_curves.png', bbox_inches='tight')
    plt.close()
    
    ##########################################################################
    
    # Generate plot with the accuracies
    fig, ax = plt.subplots(dpi=100, figsize=(8,6), nrows=1, ncols=1)
    ax.set_xscale('log')
    ax.set_xlabel(r'$t_{\mathrm{G}}$ [MCMC steps]', size=15)
    ax.set_ylabel('Accuracy', size=20)
    ax.set_ylim(bottom=None, top=1.01)
    for i, t_age in enumerate(t_ages):
        ax.plot(record_times, accuracies[i], c=colors(i), lw=3, label=r'$t_{\mathrm{age}}=$' + str(t_age * UpdByEpoch))
    
    ncol = (n_curves // 2)
    ax.legend(bbox_to_anchor=(0.5, 1.), loc="lower center", fontsize=12, ncol=ncol)
    plt.subplots_adjust(right=0.95)
    fig.text(1, 0.5, '$\mathrm{' + args.model.name.replace('_', '\_') + '}$', ha='right', va='center', rotation=90, size=5)
    fig.savefig(output / 'accuracy_curves.svg')
    fig.savefig(output / 'accuracy_curves.png', bbox_inches='tight')
    plt.close()
    
    ##########################################################################
    
    # Generate confusion matrices using the maximum generation time
    for i, t_age in enumerate(t_ages):
        cmat = confusion_matrix(targets_prediction.cpu().numpy(), predicted_labels[i].numpy(), normalize='true')
        fig, ax = plt.subplots(dpi=100, nrows=1, ncols=1)
        im = ax.imshow(cmat)
        ax.set_title(r'$t_{\mathrm{age}}=$' + str(t_age * UpdByEpoch), size=15)
        ax.set_ylabel('True labels', size=15)
        ax.set_xlabel('Predicted labels', size=15)
        ax.set_xticks(unique_labels)
        ax.set_yticks(unique_labels)
        ax.set_yticklabels([category2label[n].replace('_', '\_') for n in unique_labels], rotation=45, ha='right');
        ax.set_xticklabels([category2label[n].replace('_', '\_') for n in unique_labels], rotation=45, ha='right');
        plt.colorbar(im, label="Fraction of data", orientation="vertical")
        fig.savefig(output / f'confusion_matrix_age_{t_age}.svg')
        fig.savefig(output / f'confusion_matrix_age_{t_age}.png', bbox_inches='tight')
        plt.close()
       
    ####################################################################
    
    # Generate the plots with copy rates
    
    fig = plt.figure(dpi=100)
    plt.xlabel('age (epochs)', size=15)
    plt.ylabel('copy rate', size=15)
    plt.title('Intra-copy distributions', size=15)
    plt.grid(ls='dashed', zorder=0)
    boxplot(intra_copy_distributions, palette='hls', medianprops={'color' : 'crimson', 'linewidth' : 2}, width=0.5, flierprops={'marker': 'o'})
    plt.xticks(np.arange(len(t_ages)), t_ages)
    fig.savefig(output / 'intra_copy_boxplot.svg')
    fig.savefig(output / 'intra_copy_boxplot.png', bbox_inches='tight')
    plt.close()
    
    fig = plt.figure(dpi=100)
    plt.xlabel('age (epochs)', size=15)
    plt.ylabel('copy rate', size=15)
    plt.title('Inter-copy distributions', size=15)
    plt.grid(ls='dashed', zorder=0)
    boxplot(inter_copy_distributions, palette='hls', medianprops={'color' : 'crimson', 'linewidth' : 2}, width=0.5, flierprops={'marker': 'o'})
    plt.xticks(np.arange(len(t_ages)), t_ages)
    fig.savefig(output / 'inter_copy_boxplot.svg')
    fig.savefig(output / 'inter_copy_boxplot.png', bbox_inches='tight')
    plt.close()

    ###################################################################

    # Generate the plot of the eigenvalues

    fig, ax = plt.subplots(dpi=100, nrows=1, ncols=2, figsize=(10, 4))

    ax[0].set_title('W\'s eigenvalues', size=15)
    ax[0].set_xlabel('epoch', size=12)
    ax[0].set_ylabel('value', size=12)
    ax[0].grid(alpha=0.5, ls='dashed')
    ax[0].loglog(alltime, alls_W.t());
    ax[1].set_title('D\'s eigenvalues', size=15)
    ax[1].set_xlabel('epoch', size=12)
    ax[1].set_ylabel('value', size=12)
    ax[1].grid(alpha=0.5, ls='dashed')
    ax[1].loglog(alltime, alls_D.t());
    fig.savefig(output / 'eigenvalues.svg')
    fig.savefig(output / 'eigenvalues.png', bbox_inches='tight')
    plt.close()
    
    # save data into files
    logger.info('Saving scores into a file')
    
    f = File(scores_dir / 'energy_score.h5', 'w')
    f['record_times'] = record_times
    for t_age in t_ages:
        f[str(t_age * UpdByEpoch)] = score_LL.get_mean_score(t_age)
    f.close()
    
    f = File(scores_dir / 'entropy_score.h5', 'w')
    f['record_times'] = record_times
    for t_age in t_ages:
        f[str(t_age * UpdByEpoch)] = score_entropy.get_mean_score(t_age)
    f.close()
    
    f = File(scores_dir / 'AAI_score.h5', 'w')
    f['record_times'] = record_times
    for t_age in t_ages:
        f[str(t_age * UpdByEpoch)] = score_AAI.get_mean_score(t_age)
    f.close()
    
    f = File(scores_dir / 'AAI_data_score.h5', 'w')
    f['record_times'] = record_times
    for t_age in t_ages:
        f[str(t_age * UpdByEpoch)] = score_AAI_data.get_mean_score(t_age)
    f.close()
    
    f = File(scores_dir / 'AAI_gen_score.h5', 'w')
    f['record_times'] = record_times
    for t_age in t_ages:
        f[str(t_age * UpdByEpoch)] = score_AAI_gen.get_mean_score(t_age)
    f.close()
    
    f = File(scores_dir / 'spectra_score.h5', 'w')
    f['record_times'] = record_times
    for t_age in t_ages:
        f[str(t_age * UpdByEpoch)] = score_spectra.get_mean_score(t_age)
    f.close()
    
    f = File(scores_dir / 'first_moment_score.h5', 'w')
    f['record_times'] = record_times
    for t_age in t_ages:
        f[str(t_age * UpdByEpoch)] = score_first_moment.get_mean_score(t_age)
    f.close()
    
    f = File(scores_dir / 'second_moment_score.h5', 'w')
    f['record_times'] = record_times
    for t_age in t_ages:
        f[str(t_age * UpdByEpoch)] = score_second_moment.get_mean_score(t_age)
    f.close()
    
        
    t_stop = time.time()
    
    logger.info(f'Completed: evaluation took {round((t_stop - t_start) / 60, 1)} minutes')