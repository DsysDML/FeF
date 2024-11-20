import matplotlib.pyplot as plt

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