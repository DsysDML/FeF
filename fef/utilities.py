import logomaker
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
import readline
import select
import torch
from torch.nn.functional import one_hot

def word_to_number(string):
    amino_letters = '-GALMFWKQESPVICYHRNDT'
    letter_map = {l : n for l,n in zip(amino_letters, range(21))}
    n_list = []
    for l in string:
        n_list.append(letter_map[l])
    return n_list

def number_to_word(X):
    """X is a single numeric sequence"""
    
    amino_letters = '-GALMFWKQESPVICYHRNDT'
    seq = []
    for aa in X:
        seq.append(amino_letters[aa])
    return ''.join(seq)

def word_to_number_RNA(string):
    basis_letters = '-AGCU'
    letter_map = {l : n for l,n in zip(basis_letters, range(5))}
    n_list = []
    for l in string:
        n_list.append(letter_map[l])
    return n_list

def number_to_word_RNA(X):
    """X is a single numeric sequence"""
    
    basis_letters = '-AGCU'
    seq = []
    for aa in X:
        seq.append(basis_letters[aa])
    return ''.join(seq)

def plot_logo(logo_matrix, ax, msa=False):
    col_names = [l for l in '-GALMFWKQESPVICYHRNDT']
    if msa:
        prob_matrix = one_hot(logo_matrix).float().mean(-1)
    else:
        prob_matrix = logo_matrix
    site_entropy = -np.sum(prob_matrix * np.log2(prob_matrix), 1)
    conservation_score = (np.log2(21) - site_entropy).reshape(len(site_entropy), 1)
    conservation_matrix = prob_matrix * conservation_score
    logo_matrix = pd.DataFrame(data=conservation_matrix, columns=col_names)
    logo_matrix.index.name = 'pos'
    logo_matrix.index += 1
    
    # create Logo object
    crp_logo = logomaker.Logo(logo_matrix,
                              ax=ax,
                              shade_below=.5,
                              font_name='DejaVu Sans',
                              color_scheme='skylign_protein')

    # style using Logo methods
    crp_logo.style_spines(visible=False)
    crp_logo.style_spines(spines=['left', 'bottom'], visible=True)
    crp_logo.style_xticks(rotation=90, fmt='%d', anchor=0)

    # style using Axes methods
    crp_logo.ax.set_ylabel("Conservation (bits)", size=12)
    crp_logo.ax.set_xlabel("site", size=12)
    crp_logo.ax.xaxis.set_ticks_position('none')
    crp_logo.ax.xaxis.set_tick_params(pad=-1)

def catch_file(main_repo : str, message : str) -> str:
    """Allows to dynamically navigate among the files with text autocompletion.
    
    Parameters:
        main_repo (str): Repository to start with.
        message (str): Message to print when asking to insert the filename.
        
    Returns:
        str: selected filename.
    """
    current_repo = os.getcwd()
    os.chdir(main_repo)
    
    def completer(text, state):
        dir_path = '/'.join([dir for dir in text.split('/')[:-1]])
        filename = text.split('/')[-1]
        if dir_path != '':
            files = os.listdir('./' + dir_path)
        else:
            files = os.listdir('.' + dir_path)
        matches = [f for f in files if f.startswith(filename)]
        if dir_path != '':
            try:
                return dir_path + '/' + matches[state]
            except IndexError:
                return None
        else:
            try:
                return matches[state]
            except IndexError:
                return None
   
    # Enable name completion
    readline.parse_and_bind('tab: complete')
    # Set the completer function
    readline.set_completer(completer)
    # Set the delimiters
    readline.set_completer_delims(' \t\n')
    filename = input(message)
    
    os.chdir(current_repo)
    
    return main_repo + '/' + filename

def select_device():
    num_devices = torch.cuda.device_count()
    if num_devices > 0:
        print(f'Found n.{num_devices} cuda devices.')
        if num_devices == 1:
            print('cuda device automatically selected.')
            device = torch.device('cuda')
        else:
            dev_idx = input(f'Select the cuda device {[i for i in range(num_devices)]}: ')
            device = torch.device(f'cuda:' + dev_idx)
    else:
        device = torch.device('cpu')
    
    return device

def check_if_exists(fname):
    if os.path.exists(fname):
        good_input = False
        overwrite = 'y'
        while not good_input:
            print(f'The file {fname} already exists. Do you want to overwrite it? [y/n] ')
            i, _, _ = select.select([sys.stdin], [], [], 10) # 10 seconds to answer
            if i:
                overwrite = sys.stdin.readline().strip()
            if overwrite in ['n', 'N']:
                sys.exit('Execution aborted')
            elif overwrite in ['y', 'Y']:
                os.remove(fname)
                good_input = True
                
def import_from_fasta(fasta_name):
    sequences = []
    names = []
    seq = ''
    with open(fasta_name, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            if line.startswith('>'):
                if seq:
                    sequences.append(seq)
                header = line[1:].strip().replace(' ', '_')
                names.append(header)
                seq = ''
            else:
                seq += line.strip()
    if seq:
        sequences.append(seq)
    
    return np.array(names, dtype='U'), np.array(sequences)

def write_fasta(fname, headers, sequences, numeric_input=False, remove_alignment=False):
    if numeric_input:
        with open(fname, 'w') as f:
            for name_seq, num_seq in zip(headers, sequences):
                f.write('>' + name_seq + '\n')
                if remove_alignment:
                    f.write(number_to_word(num_seq[num_seq != 0]))
                else:
                    f.write(number_to_word(num_seq))
                f.write('\n')
    else:
        with open(fname, 'w') as f:
            for name_seq, seq in zip(headers, sequences):
                f.write('>' + name_seq + '\n')
                f.write(seq.replace('X', ''))
                f.write('\n')

def generate_iTOL_annotation_file(fname, names, labels, fun2color):
    with open(fname, 'w') as f:
        f.write('TREE_COLORS\n')
        f.write('SEPARATOR TAB\n')
        f.write('DATA\n')
        for name, fun in zip(names, labels):
            f.write('{0}\trange\t{1}\t{2}\n'.format(name, fun2color[fun], fun))