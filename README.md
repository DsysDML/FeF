# F&F - Fast and Functional Structured Data Generators
Code for the paper "Fast and Functional Structured Data Generators Rooted in Out-of-Equilibrium Physics" by Alessandra Carbone, Aurélien Decelle, Lorenzo Rosset and Beatriz Seoane.

## Installation
All the dependencies needed to run the code in this repository can be installed using `pip` though
```bash
pip install -r requirements.txt
``` 
or, using `conda`,
```bash
conda install --yes --file requirements.txt
```
To set up the environment variables, add the following lines to the .bashrc file:
```bash
export PATH=${PATH}:/installation_path/RBM-Proteins
export RBMHOME=/installation_path/RBM-Proteins
```

## Example data
To download an example of an input file for training the RBM using the MNIST dataset, use
```bash
wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=1XiP_KPKuGZmxoqQz6tnVqUFlxf44S5kX' -O 'data/MNIST.h5'
```

## Training RBMs

### Dataset format
Datasets are handled through the `.h5` format. The following fields are needed:

Mandatory:
- *train*: training set data;
- *train_labels*: labels for the training data in string format. If no label is available for specific data points, use the dummy label '-1';

Optional:
- *train_names*: train set names for the tree leaves;
- *test*: test set data;
- *test_labels*: labels for the test data in string format. If no label is available for specific data points, use the dummy label '-1';
- *test_names*: test set names for the tree leaves.


### Launching a training
The basic syntax for starting the training of an RBM model is
```bash
./rbm-train.sh -m <model_type> -d <path_to_data>
```
The list of available models can be printed through
```bash
./rbm-train.sh -h
```
and it consists of all the classes in the repository `src/RBMs` without the suffix `RBM.py`.

In order to know all the available training options for a particular type of RBM model, use
```bash
./rbm-train.sh -m <model_type> -h
```
The Python files that handle the training of the models are contained in `src/train`.
