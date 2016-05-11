from __future__ import print_function
import six.moves.cPickle as pickle

from collections import OrderedDict
import h5py
import numpy as np
import theano
import theano.tensor as tensor
from theano import config
from theano.tensor.signal.pool import pool_2d
from theano.tensor.nnet.conv import conv2d
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from collections import OrderedDict


model_path = '/Users/liyulin/Documents/python/BiLSTM/model/'
model_name = 'batch_64_pretrain_no_dropout_no_blur_14_x4.npz'
model = np.load(model_path + model_name)


