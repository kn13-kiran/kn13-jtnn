# Training of Junction Tree VAE
Suppose the repository is downloaded at `$PREFIX/kn13-jtnn` directory. First set up environment variables:
```
export PYTHONPATH=$PREFIX/kn13-jtnn
```
The MOSES dataset is in `kn13-jtnn/data/moses` (copied from https://github.com/molecularsets/moses).

## STEP1: Building Vocabulary set 
This step extracts the molecules from smiles and converts them to a dictionary that can be used for embeddings.
If you are running this code on a new dataset, you need to compute the vocabulary from your dataset.
To perform tree decomposition over a set of molecules, run
```
python ../model/molecule_tree.py < ../data/moses/sample.txt
python ../model/molecule_tree.py < ../data/moses/train.txt
```
This gives you the vocabulary of cluster labels over the dataset `sample.txt` or `train.txt`. 
This can be saved to `vocab-sample.txt` which is used for training.
For a local machine (non-gpu run), use the `sample.txt`.
Instead of re-generating, you can also re-use the existing vocabulary coming from the training data set from the original repository.

## STEP2: Preprocess Molecules and generate tensors
```
python preprocess.py --train ../data/moses/sample.txt --split 100 --jobs 16
python preprocess.py --train ../data/moses/train.txt --split 100 --jobs 16

mkdir moses-processed-sample-r1
mv tensor* moses-processed-sample-r1
```
This script will preprocess the training data (subgraph enumeration & tree decomposition), and save results into a list of files.
For smaler datasets, it is preferred to use We suggest you to use small value for `--split` parameter.

## STEP3: Training

Train JunctionTreeVariationalEncoder model with KL annealing.
The following step generates the model to `vae_model_sample-r1`
```
mkdir vae_model/
python vae_train.py --train moses-processed --vocab ../data/vocab-sample.txt --save_dir vae_model_sample-r1/
```
Default Options:

`--beta 0` means to set KL regularization weight (beta) initially to be zero.

`--warmup 40000` means that beta will not increase within first 40000 training steps. It is recommended because using large KL regularization (large beta) in the beginning of training is harmful for model performance.

`--step_beta 0.002 --kl_anneal_iter 1000` means beta will increase by 0.002 every 1000 training steps (batch updates). You should observe that the KL will decrease as beta increases.

`--max_beta 1.0 ` sets the maximum value of beta to be 1.0. 

`--save_dir vae_model`: the model will be saved in vae_model_sample-r1/

Please note that this is not necessarily the best annealing strategy. You are welcomed to adjust these parameters.

## Step4: Testing (Generating sample molecules)
To sample new molecules with trained models, simply run
```
python molecule_generator.py --nsample 30000 --vocab ../data/moses/vocab-sample.txt --hidden 450 --model vae_model_sample-r1/model.iter-5000> sample30k.txt
```
This script prints in each line the SMILES string of each molecule. `model.iter-5000` is a model trained with 4k steps (model built on local machine with sample vocabulary) with the default hyperparameters.
This should give you the same samples as in `sample30k.txt`.

## Step4: Testing molecule reconstruction
You can try to reconstruct the original molecules using two options:
Option1: Using the model `model.iter-5000`.
```
python rebuild.py --test ../data/moses/vocab-sample.txt  --vocab ../data/moses/vocab-sample.txt --hidden 450 --depth 3 --latent 56  --model vae_model_sample-r1/model.iter-5000
```
Option2: Using the pre-trianed model from the original paper's implementation.
```
python reconstruct.py --test ../data/zinc/valid.txt --vocab ../data/zinc/vocab.txt --hidden 450 --depth 3 --latent 56 --model MPNVAE-h450-L56-d3-beta0.005/model.iter-4
