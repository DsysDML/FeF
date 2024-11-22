import matplotlib.pyplot as plt
from fef.scores import Score
import numpy as np

def plot_scores_mean(
    filename: str,
    score_ll: Score,
    score_spectrum: Score,
    score_AAI_data: Score,
    score_AAI_gen: Score,
    score_entropy: Score,
    score_first_moment: Score,
    score_second_moment: Score,
) -> None:
    
    num_curves = len(score_ll.checkpoints)
    colors = plt.get_cmap('RdYlBu', num_curves)
    fig, ax = plt.subplots(6, 1, sharex=True, dpi=192, figsize=(6, 16))
    
    ax[0].set_ylabel(r'$\epsilon^{\mathrm{LL}}$', size=15)
    ax[0].set_xscale('log')
    ax[0].set_yscale('log')
    
    ax[1].set_ylabel(r'$\epsilon^{\mathrm{s}}$', size=15)
    ax[1].set_xscale('log')
    ax[1].set_yscale('log')
    
    ax[2].set_ylabel(r'$\Delta S$', size=15)
    ax[2].set_xscale('log')
    ax[2].set_yscale('log')
    
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
    
    record_times = score_ll.record_times
    for i, checkpoint in enumerate(score_ll.checkpoints):
        ax[0].plot(record_times, score_ll.mean_across_labels()[i], label=r'$t_{\mathrm{age}}=$' + str(checkpoint), c=colors(i))
        ax[1].plot(record_times, score_spectrum.mean_across_labels()[i], c=colors(i))
        ax[2].plot(record_times, score_entropy.mean_across_labels()[i], c=colors(i))
        data_AAI = score_AAI_data.mean_across_labels()[i]
        gen_AAI = score_AAI_gen.mean_across_labels()[i]
        AAI_score = 0.5 * ((data_AAI - 0.5)**2 + (gen_AAI - 0.5)**2)
        ax[3].plot(record_times, AAI_score, c=colors(i))
        ax[4].plot(record_times, score_first_moment.mean_across_labels()[i], c=colors(i))
        ax[5].plot(record_times, score_second_moment.mean_across_labels()[i], c=colors(i))
    
    ncol = num_curves // 2 if num_curves > 1 else 1
    ax[0].legend(bbox_to_anchor=(0.5, 1.8), loc="upper center", fontsize=12, ncol=ncol)
    
    plt.subplots_adjust(right=0.95)
    fig.savefig(filename, bbox_inches='tight')
    plt.close()
    
    
def plot_eigenvalues(
    filename: str,
    updates: list | np.ndarray,
    eigenvalues: list | np.ndarray,
    eigenvalues_labels: list | np.ndarray,
) -> None:
    fig, ax = plt.subplots(2, 1, figsize=(6, 4), sharex=True, gridspec_kw={'hspace': 0})
    ax[0].plot(updates, eigenvalues, lw=1)
    ax[1].plot(updates, eigenvalues_labels, lw=1)
    ax[1].set_xlabel("Training time (updates)")
    ax[0].set_ylabel("Weight matrix")
    ax[1].set_ylabel("Label matrix")
    ax[0].set_title("Eigenvalues history")
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    
    
def plot_AAI_scores(
    filename: str,
    score_AAI_data: Score,
    score_AAI_gen: Score,
) -> None:
    fig, ax = plt.subplots(dpi=192, figsize=(15,6), nrows=1, ncols=2)
    checkpoints = score_AAI_data.checkpoints
    record_times = score_AAI_data.record_times
    num_curves = len(checkpoints)
    colors = plt.get_cmap('RdYlBu', num_curves)
    
    ax[0].set_xscale('log')
    ax[0].set_xlabel(r'$t_{\mathrm{G}}$ [MCMC steps]', size=15)
    ax[0].set_ylabel('AAI data', size=20)
    ax[0].axhline(y=0.5, ls='dashed', c='black')
    for i, checkpoint in enumerate(checkpoints):
        ax[0].plot(record_times, score_AAI_data.mean_across_labels()[i], c=colors(i), lw=3, label=r'$t_{\mathrm{age}}=$' + str(checkpoint))
    lines, labels = ax[0].get_legend_handles_labels()
        
    ax[1].set_xscale('log')
    ax[1].set_xlabel(r'$t_{\mathrm{G}}$ [MCMC steps]', size=15)
    ax[1].set_ylabel('AAI generated', size=20)
    ax[1].axhline(y=0.5, ls='dashed', c='black')
    for i, checkpoint in enumerate(checkpoints):
        ax[1].plot(record_times, score_AAI_gen.mean_across_labels()[i], c=colors(i), lw=3, label=r'$t_{\mathrm{age}}=$' + str(checkpoint))
        
    ncol = num_curves // 2 if num_curves > 1 else 1
    fig.legend(lines, labels, bbox_to_anchor=(0.5, 1.), loc="upper center", fontsize=12, ncol=ncol)
    plt.subplots_adjust(right=0.95)
    fig.savefig(filename, bbox_inches='tight')
    plt.close()
    
    
def plot_accuracies(
    filename: str,
    score_accuracy: Score,
) -> None:
    record_times = score_accuracy.record_times
    checkpoints = score_accuracy.checkpoints
    num_curves = len(checkpoints)
    colors = plt.get_cmap('RdYlBu', num_curves)
    fig, ax = plt.subplots(dpi=192, figsize=(8,6), nrows=1, ncols=1)
    ax.set_xscale('log')
    ax.set_xlabel(r'$t_{\mathrm{G}}$ [MCMC steps]', size=15)
    ax.set_ylabel('Accuracy', size=20)
    ax.set_ylim(bottom=None, top=1.01)
    for i, checkpoint in enumerate(checkpoints):
        ax.plot(record_times, score_accuracy.mean_across_labels()[i], c=colors(i), lw=3, label=r'$t_{\mathrm{age}}=$' + str(checkpoint))
    
    ncol = num_curves // 2 if num_curves > 1 else 1
    ax.legend(bbox_to_anchor=(0.5, 1.), loc="lower center", fontsize=12, ncol=ncol)
    plt.subplots_adjust(right=0.95)
    fig.savefig(filename, bbox_inches='tight')
    plt.close()
    
    
def plot_confusion_matrix(
    filename: str,
    confusion_matrix: np.ndarray,
    checkpoint: int,
    labels: list,
) -> None:
    
    fig, ax = plt.subplots(dpi=192, nrows=1, ncols=1)
    im = ax.imshow(confusion_matrix, cmap='viridis')
    ax.set_title(r'$t_{\mathrm{age}}=$' + str(checkpoint), size=15)
    ax.set_ylabel('True labels', size=15)
    ax.set_xlabel('Predicted labels', size=15)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, rotation=45, ha='right');
    ax.set_xticklabels(labels, rotation=45, ha='right');
    plt.colorbar(im, label="Fraction of data", orientation="vertical")
    fig.savefig(filename, bbox_inches='tight')
    plt.close()
    