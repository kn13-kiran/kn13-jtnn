import torch
import torch.nn as nn

import math, random, sys
import argparse
from model import *
import rdkit
import rdkit.Chem as Chem

lg = rdkit.RDLogger.logger() 
lg.setLevel(rdkit.RDLogger.CRITICAL)

parser = argparse.ArgumentParser()
parser.add_argument('--nsample', type=int, required=True)
parser.add_argument('--vocab', required=True)
parser.add_argument('--model', required=True)

parser.add_argument('--hidden_size', type=int, default=450)
parser.add_argument('--latent_size', type=int, default=56)
parser.add_argument('--depthT', type=int, default=20)
parser.add_argument('--depthG', type=int, default=3)

args = parser.parse_args()
   
vocab = [x.strip("\r\n ") for x in open(args.vocab)] 
vocab = Vocab(vocab)

model = JunctionTreeVariationalEncoder(vocab, args.hidden_size, args.latent_size, args.depthT, args.depthG)
model.load_state_dict(torch.load(args.model))

#set the seed for random number generator
torch.manual_seed(0)

acc=0
gen=0
for i in xrange(args.nsample):
    smiles= model.sample_prior()
    print(smiles)
    gen +=1
    try:
        mol = Chem.MolFromSmiles(smiles)
        smiles3D = Chem.MolToSmiles(mol, isomericSmiles=True)
        acc+=1
    except:
        print('Invalid smiles')
print("Sample generation Acuracy",(gen/acc))