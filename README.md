# equivariant_electron_density
Generate and predict molecular electron densities with Euclidean Neural Networks

Below is a workflow for how to use the scripts in this repository.
## Step 1: Generate molecular electron densities from a set of input coordinates.
The only neccesary inputs are a atomic coordinates file and an auxiliary basis set file. You can find examples of both in `tests/test_data_generation`. Atomic coordinates must be supplied in [XYZ file format](https://en.wikipedia.org/wiki/XYZ_file_format).

> Command: `python densityfit_q.py xyz_filename orbital_basis density_fit_basis`
> 
> Example: `python densityfit_q.py water.xyz aug-cc-pvtz def2-universal-jfit-decontract`

This command uses the `densityfit_q.py` script to run a quantum chemistry calculation with the `psi4` quantum chemistry program. Then it projects the electron density onto the density fitting basis. This will produce a psi4 output file (from which one can extract energy and forces) and density output file with the coefficients of the density fitting basis set.


## Step 2: Create the dataset
We must now parse the output files to create a dataset for training. This is done with the `create_dataset.py` script.

> Command: `python create_dataset.py path/to/data/folder dataset_name`
> 
> Example: `python create_dataset.py ../tests/test_data_generation water_density_dataset.pkl`

The dataset will now be the input we use to train our `e3nn` network.

## Step 3: Train the model
Now it's time to train an `e3nn` model on our dataset. We will use the `train_density.py` script in `training` to do this. There are a number of keyword arguments to `train_density.py`.

- "dataset": path to training dataset
- "testset": path to test dataset
- "split": number of samples from the dataset to use for training
- "epochs": number of epochs for training

> Command: `python train_density.py --dataset path/to/dataset --testset path/to/testset --split n_samples --epochs n_epochs`
> 
> Example: `python train_density.py --dataset ../tests/water_density_dataset.pkl --testset ../tests/water_density_testset.pkl --split 100 --epochs 500`

The script is set up to track training and test metrics in `wandb`, so you'll need an account to see how training is going.


For additional resources, see the [e3nn tutorial](https://e3nn.org/e3nn-tutorial-mrs-fall-2021/). Check out the tutorial on electron densities [here](https://colab.research.google.com/drive/1ryOQ6hXxCidM_mGN0Yrf4BbjUtpyCxgy#scrollTo=PTTwyYkhioyc)
