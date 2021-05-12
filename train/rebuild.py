import torch
import torch.nn as nn
from torch.autograd import Variable

import math, random, sys
from optparse import OptionParser
from collections import deque

import rdkit
import rdkit.Chem as Chem

from model import *
import time

lg = rdkit.RDLogger.logger()
lg.setLevel(rdkit.RDLogger.CRITICAL)

parser = OptionParser()
parser.add_option("-t", "--test", dest="test_path")
parser.add_option("-v", "--vocab", dest="vocab_path")
parser.add_option("-m", "--model", dest="model_path")
parser.add_option("-w", "--hidden", dest="hidden_size", default=200)
parser.add_option("-l", "--latent", dest="latent_size", default=56)
parser.add_option("-d", "--depth", dest="depth", default=3)
parser.add_option("-e", "--stereo", dest="stereo", default=1)
opts, args = parser.parse_args()

vocab = [x.strip("\r\n ") for x in open(opts.vocab_path)]
vocab = Vocab(vocab)

hidden_size = int(opts.hidden_size)
latent_size = int(opts.latent_size)
depth = int(opts.depth)
stereo = True if int(opts.stereo) == 1 else False

model = JunctionTreeVariationalEncoder(vocab, hidden_size, latent_size, depth, depth)


model.load_state_dict(torch.load(opts.model_path, map_location=torch.device('cpu')))

data = []
with open(opts.test_path) as f:
    for line in f:
        s = line.strip("\r\n ").split()[0]
        data.append(s)

acc = 0.0
tot = 0
#time.sleep(10)

for smiles in data:
    mol = Chem.MolFromSmiles(smiles)
    smiles3D = Chem.MolToSmiles(mol, isomericSmiles=True)
    dec_smiles = model.reconstruct(smiles)
    #print(smiles,dec_smiles,smiles3D)
    if dec_smiles == smiles3D:
        acc += 1
    tot += 1
    print acc / tot
