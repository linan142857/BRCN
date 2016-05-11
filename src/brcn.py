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


"""
a random number generator used to initialize weights
"""
SEED = 13455
rng = np.random.RandomState(SEED)
np.random.seed(SEED)

class Scan(object):
    """
    Scan the time-seq for RNN

    Scanning is a general form of recurrence, which can be used for looping.
    The idea is that you *scan* a function along some input sequence, producing
    an output at each time-step that can be seen (but not modified) by the
    function at the next time-step. (Technically, the function can see the
    previous K time-steps of your outputs and L time steps (from past and
    future) of your inputs.

    """

    def __init__(self,
        fn,  # a function that describes the operations
             #  involved in one step of ``scan``.
        sequences=None,  # the list of Theano variables or dictionaries
                         #  describing the sequences ``scan`` has to iterate over.
        outputs_info=None,  # the list of Theano variables or dictionaries
                            # describing the initial state of the outputs computed recurrently.
        non_sequences=None,  # the list of arguments that are passed to ``fn`` at each steps.
        n_steps=None,  # the number of steps to iterate given as an int or Theano scalar.
        go_backwards=False,  # a flag indicating if ``scan`` should go backwards through the sequences.
        name=None  # the instance appears in those profiles
                   #  and can greatly help to disambiguate information.
        ):

        seq = self._wrap_into_list(sequences)
        outs_info = self._wrap_into_list(outputs_info)
        non_seqs = []
        for elem in self._wrap_into_list(non_sequences):
            non_seqs.append(elem)

        self.fn = fn
        self.sequences = seq
        self.outputs_info = outs_info
        self.non_sequences = non_seqs
        self.n_steps = n_steps
        self.name = name
        self.go_backwards = go_backwards

        if self.n_steps is None:
            self.n_steps = self.sequences[0].shape[0]

        self.args =  self._filter(self.outputs_info) + \
                    self.non_sequences
        self.output = []


    def _filter(self, x):
        """
        Ignore or discard all input elements of 'None'.

        """
        y = []
        for i in range(len(x)):
            if self.outputs_info[i] is not None:
                y.append(x[i])
        return y

    def _wrap_into_list(self, x):
        """
        Wrap the input into a list if it is not already a list.

        """
        if x is None:
            return []
        elif not isinstance(x, (list, tuple)):
            return [x]
        else:
            return list(x)


    def scan(self):
        """

        This module provides the Scan Op.

        """
        successive_outputs = [[] for ll in self.outputs_info]
        indices = list(range(self.n_steps))
        if self.go_backwards:
                indices = indices[::-1]
        for i in indices:
            input = [ll[i, :] for ll in self.sequences] + self.args
            temp = self.fn(*input)

            for j in range(len(successive_outputs)):
                successive_outputs[j].append(temp[j])

            self.args = self._filter(temp) + self.non_sequences
        for i in successive_outputs:
            self.output.append(tensor.stack(*i))
        return self.output

class ConvLayer(object):
    """
    Pool Layer of a convolutional network
    """

    def __init__(self, filter_shape, W=None, b=None):
        """
        Allocate a c with shared variable internal parameters.

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height, filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows, #cols)

        :type name: str
        :param name: given a special name for the ConvPoolLayer
        """

        # self.filter_shape = filter_shape
        # self.image_shape = image_shape
        # self.poolsize = poolsize

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        # fan_in = np.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        # fan_out = (filter_shape[0] * np.prod(filter_shape[2:]) //
        #            np.prod(poolsize))
        # initialize weights with random weights

        # W_bound = np.sqrt(6. / (fan_in + fan_out))
        if W is not None:
            self.W = theano.shared(W, borrow=True)
        else:
            self.W = theano.shared(
                np.asarray(
                    rng.normal(0, 0.001, size=filter_shape),
                    dtype=config.floatX
                ),
                borrow=True
            )

        # the bias is a 1D tensor -- one bias per output feature map
        if b is not None:
            self.b = theano.shared(b, borrow=True)
        else:
            self.b = theano.shared(np.zeros(filter_shape[0]).astype(config.floatX), borrow=True)

        # store parameters of this layer
        self.params = [self.W, self.b]

    def conv(self, input):
        # convolve input feature maps with filters
        conv_out = conv2d(
            input=input,
            filters=self.W
        )

        # downsample each feature map individually, using maxpooling
        # pooled_out = pool_2d(
        #     input=conv_out,
        #     ds=self.poolsize,
        #     ignore_border=True
        # )

        # output = pooled_out + self.b.dimshuffle('x', 0, 'x', 'x')
        output = conv_out + self.b.dimshuffle('x', 0, 'x', 'x')

        return output

class BiRecConvNet(object):
    """Bidirectional Recurrent Convolutional Networks"""

    def __init__(self, options):
        self.options = options

    def build_net(self, model=None):
        options = self.options
        if model is None:
            model = [None, None]
        # forward & backward data flow
        x = tensor.TensorType(dtype=config.floatX, broadcastable=(False,) * 5)(name='x')
        y = tensor.TensorType(dtype=config.floatX, broadcastable=(False,) * 5)(name='y')

        # forward net
        layers_f, params_f = self._init_layer(options['size'], options['filter_shape'], options['rec_filter_size'], model[0])
        proj_f, use_noise_f = self._build_model(x, options, layers_f, params_f, go_backwards=False)

        # backward net
        layers_b, params_b = self._init_layer(options['size'], options['filter_shape'], options['rec_filter_size'], model[1])
        proj_b, use_noise_b = self._build_model(x, options, layers_b, params_b, go_backwards=True)


        proj = proj_f + proj_b[::-1]

        cost = ((y - proj) ** 2).sum() / (x.shape[1] * x.shape[0])

        f_x = theano.function([x], proj, name='f_proj')
        params = dict(prefix_p('f', params_f), **(prefix_p('b', params_b)))

        return x, y, f_x, cost, params, use_noise_f, use_noise_b

    def _init_layer(self, size, filter_shape, rec_filter_size, model):
        """
        Global (net) parameter. For the convolution and regular opt.
        """
        layers = OrderedDict()
        params = OrderedDict()
        if model is None:
            model = dict()

        ''' for layer1 '''

        layers['conv_1_v'] = ConvLayer(filter_shape[0], model.get('conv_1_v_w'), model.get('conv_1_v_b'))
        layers['conv_1_r'] = ConvLayer(rec_filter_size[0], model.get('conv_1_r_w'), model.get('conv_1_r_b'))
        layers['conv_1_t'] = ConvLayer(filter_shape[0], model.get('conv_1_t_w'), model.get('conv_1_t_b'))
        params['conv_1_v_w'] = layers['conv_1_v'].params[0]
        params['conv_1_v_b'] = layers['conv_1_v'].params[1]
        params['conv_1_r_w'] = layers['conv_1_r'].params[0]
        params['conv_1_r_b'] = layers['conv_1_r'].params[1]
        params['conv_1_t_w'] = layers['conv_1_t'].params[0]
        params['conv_1_t_b'] = layers['conv_1_t'].params[1]
        if model.get('b_1') is not None:
            params['b_1'] = theano.shared(model['b_1'].astype(config.floatX), name='b_1', borrow=True)
        else:
            params['b_1'] = theano.shared(np.zeros(filter_shape[0][0]).astype(config.floatX), name='b_1', borrow=True)

        ''' for layer2 '''

        layers['conv_2_v'] = ConvLayer(filter_shape[1], model.get('conv_2_v_w'), model.get('conv_2_v_b'))
        layers['conv_2_r'] = ConvLayer(rec_filter_size[1], model.get('conv_2_r_w'), model.get('conv_2_r_b'))
        layers['conv_2_t'] = ConvLayer(filter_shape[1], model.get('conv_2_t_w'), model.get('conv_2_t_b'))
        params['conv_2_v_w'] = layers['conv_2_v'].params[0]
        params['conv_2_v_b'] = layers['conv_2_v'].params[1]
        params['conv_2_r_w'] = layers['conv_2_r'].params[0]
        params['conv_2_r_b'] = layers['conv_2_r'].params[1]
        params['conv_2_t_w'] = layers['conv_2_t'].params[0]
        params['conv_2_t_b'] = layers['conv_2_t'].params[1]
        if model.get('b_2') is not None:
            params['b_2'] = theano.shared(model['b_2'].astype(config.floatX), name='b_2', borrow=True)
        else:
            params['b_2'] = theano.shared(np.zeros(filter_shape[1][0]).astype(config.floatX), name='b_2', borrow=True)

        ''' for layer3 '''

        layers['conv_3_v'] = ConvLayer(filter_shape[2], model.get('conv_3_v_w'), model.get('conv_3_v_b'))
        layers['conv_3_t'] = ConvLayer(filter_shape[2], model.get('conv_3_v_w'), model.get('conv_3_t_b'))
        params['conv_3_v_w'] = layers['conv_3_v'].params[0]
        params['conv_3_v_b'] = layers['conv_3_v'].params[1]
        params['conv_3_t_w'] = layers['conv_3_t'].params[0]
        params['conv_3_t_b'] = layers['conv_3_t'].params[1]
        if model.get('b_3') is not None:
            params['b_3'] = theano.shared(model['b_3'].astype(config.floatX), name='b_3', borrow=True)
        else:
            params['b_3'] = theano.shared(np.zeros(filter_shape[2][0]).astype(config.floatX), name='b_3', borrow=True)


        return layers, params

    def _build_model(self, input, options, layers, params, go_backwards=False):
        # Used for dropout.
        trng = RandomStreams(SEED)
        use_noise = theano.shared(numpy_floatX(0.), borrow=True)
        nsteps = options['n_timestep']
        shape_1 = options['size'][0]
        shape_2 = options['size'][1]
        shape_3 = options['size'][2]
        shape_4 = options['size'][3]

        def _step(x_, t_, r_, layer_):
            layer_ = str(layer_.data)
            v = layers['conv_' + layer_ + '_v'].conv(x_)
            t = layers['conv_' + layer_ + '_t'].conv(t_)
            if layer_ != '3':
                r = layers['conv_' + layer_ + '_r'].conv(r_)
                h = tensor.nnet.relu(v + t + r + params['b_' + layer_].dimshuffle('x', 0, 'x', 'x'))
            else:
                h = v + t + params['b_' + layer_].dimshuffle('x', 0, 'x', 'x')
            return x_, h

        rval, _ = theano.scan(_step, sequences=[input],
                    outputs_info=[theano.shared(np.zeros(shape_1)
                                                .astype(config.floatX)),
                                  theano.shared(np.zeros(shape_2)
                                                .astype(config.floatX))],
                    non_sequences=[1],
                    name='rnn_layers_1',
                    n_steps=nsteps,
                    go_backwards=go_backwards)
        rval, _ = theano.scan(_step, sequences=[rval[1]],
                    outputs_info=[theano.shared(np.zeros(shape_2)
                                                .astype(config.floatX)),
                                  theano.shared(np.zeros(shape_3)
                                                .astype(config.floatX))],
                    non_sequences=[2],
                    name='rnn_layers_2',
                    n_steps=nsteps,
                    go_backwards=go_backwards)
        rval, _ = theano.scan(_step, sequences=[rval[1]],
                    outputs_info=[theano.shared(np.zeros(shape_3)
                                                .astype(config.floatX)),
                                  theano.shared(np.zeros(shape_4)
                                                .astype(config.floatX))],
                    non_sequences=[3],
                    name='rnn_layers_3',
                    n_steps=nsteps,
                    go_backwards=go_backwards)




        # def _step(x_, t1_, r1_, t2_, r2_, t3_):
        #     v1 = layers['conv_1_v'].conv(x_)
        #     t1 = layers['conv_1_t'].conv(t1_)
        #     r1 = layers['conv_1_r'].conv(r1_)
        #
        #     h1 = tensor.nnet.relu(v1 + t1 + r1 + params['b_1'].dimshuffle('x', 0, 'x', 'x'))
        #
        #     v2 = layers['conv_2_v'].conv(h1)
        #     t2 = layers['conv_2_t'].conv(t2_)
        #     r2 = layers['conv_2_r'].conv(r2_)
        #
        #     h2 = tensor.nnet.relu(v2 + t2 + r2 + params['b_2'].dimshuffle('x', 0, 'x', 'x'))
        #
        #     v3 = layers['conv_3_v'].conv(h2)
        #     t3 = layers['conv_3_t'].conv(t3_)
        #
        #     o = v3 + t3 + params['b_3'].dimshuffle('x', 0, 'x', 'x')
        #     return x_, h1, h1, h2, h2, o
        #
        #
        #
        # rval, _ = theano.scan(_step, sequences=[input],
        #             outputs_info=[theano.shared(np.zeros(shape_1)
        #                                         .astype(config.floatX)),
        #                           theano.shared(np.zeros(shape_2)
        #                                         .astype(config.floatX)),
        #                           theano.shared(np.zeros(shape_2)
        #                                         .astype(config.floatX)),
        #                           theano.shared(np.zeros(shape_3)
        #                                         .astype(config.floatX)),
        #                           theano.shared(np.zeros(shape_3)
        #                                         .astype(config.floatX)),
        #                           None
        #                           ],
        #             name='rnn_layers',
        #             n_steps=nsteps,
        #             go_backwards=go_backwards)
        proj = rval[1]
        if options['use_dropout']:
            proj = dropout_layer(proj, use_noise, trng)

        return proj, use_noise

def pred_error(f_pred, data, target, iterator, mean, max_ite=10):
    """
    Just compute the error
    f_pred: Theano fct computing the prediction
    target: usual groundtruth for that dataset.
    """
    np.random.shuffle(iterator)
    end = min(max_ite, len(iterator))

    valid_index = iterator[:end]
    psnr_err = 0.
    for v in valid_index:
        y = np.asarray([target[t, :, :, 7:27, 7:27] for t in v])
        x = np.asarray([data[t, :] for t in v])
        y = theano.shared(y).dimshuffle(1, 0, 2, 3, 4).eval()
        x = theano.shared(x).dimshuffle(1, 0, 2, 3, 4).eval()
        x = f_pred(x)
        y = np.maximum(np.minimum((y + mean) * 255, 255), 0)
        x = np.maximum(np.minimum((x + mean) * 255, 255), 0)
        psnr_err += pnsr(x, y)
    return psnr_err / len(valid_index)

def reverse_list(list, ndim=0):
    dim = range(len(list.shape.eval()))
    dim[0] = ndim - 1
    dim[ndim] = 0
    x = list.dimshuffle(dim)
    x = x[::-1]
    x = x.dimshuffle(dim)
    return x

def prefix_p(prefix, params):
    tp = OrderedDict()
    for kk, pp in params.items():
        tp['%s_%s' % (prefix, kk)] = params[kk]
    return tp

def numpy_floatX(data):
    return np.asarray(data, dtype=config.floatX)

def get_minibatches_idx(n, minibatch_size, shuffle=False):
    """
    Used to shuffle the dataset at each iteration.
    """

    idx_list = np.arange(n, dtype="int32")

    if shuffle:
        np.random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(n // minibatch_size):
        minibatches.append(idx_list[minibatch_start:
                                    minibatch_start + minibatch_size])
        minibatch_start += minibatch_size
    #
    # if (minibatch_start != n):
    #     # Make a minibatch out of what is left
    #     minibatches.append(idx_list[minibatch_start:])

    return range(len(minibatches)), minibatches

def dropout_layer(state_before, use_noise, trng):
    proj = tensor.switch(use_noise,
                         (state_before *
                          trng.binomial(state_before.shape,
                                        p=0.5, n=1,
                                        dtype=state_before.dtype)),
                         state_before * 0.5)
    return proj

def load_model(path):
    f = h5py.File(path)
    w1 = np.asarray(f.get('w1')[:], dtype=config.floatX).T
    w2 = np.asarray(f.get('w2')[:], dtype=config.floatX).T
    w3 = np.asarray(f.get('w3')[:], dtype=config.floatX).T
    b1 = np.asarray(f.get('b1')[:], dtype=config.floatX).T
    b2 = np.asarray(f.get('b2')[:], dtype=config.floatX).T
    b3 = np.asarray(f.get('b3')[:], dtype=config.floatX).T
    return [w1, w2, w3, b1, b2, b3]

def pnsr(x, y):
    z = np.mean((y - x) ** 2, axis=(2, 3, 4))
    rmse = np.sqrt(z)
    rmse = np.mean(rmse)
    psnr = 20 * np.log10(255 / rmse)
    return psnr

def load_data(path):
    """the data is scalaed in [0 1]"""

    f = h5py.File(path)
    lr_data = np.asarray(f.get('lr_data')[:], dtype=config.floatX).T
    hr_data = np.asarray(f.get('hr_data')[:], dtype=config.floatX).T
    return lr_data, hr_data

def sgd(lr, tparams, grads, x, y, cost):
    """ Stochastic Gradient Descent

    :note: A more complicated version of sgd then needed.  This is
        done like that for adadelta and rmsprop.

    """
    # New set of shared variable that will contain the gradient
    # for a mini-batch.
    gshared = [theano.shared(p.get_value() * 0., name='%s_grad' % k)
               for k, p in tparams.items()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]

    # Function that computes gradients for a mini-batch, but do not
    # updates the weights.
    f_grad_shared = theano.function([x, y], cost, updates=gsup,
                                    name='sgd_f_grad_shared')

    pup = [(p, p - lr * g) for p, g in zip(tparams.values(), gshared)]

    # Function that updates the weights from the previously computed
    # gradient.
    f_update = theano.function([lr], [], updates=pup,
                               name='sgd_f_update')

    return f_grad_shared, f_update

def adadelta(lr, tparams, grads, x, y, cost):
    """
    An adaptive learning rate optimizer

    Parameters
    ----------
    lr : Theano SharedVariable
        Initial learning rate
    tpramas: Theano SharedVariable
        Model parameters
    grads: Theano variable
        Gradients of cost w.r.t to parameres
    x: Theano variable
        Model inputs
    mask: Theano variable
        Sequence mask
    y: Theano variable
        Targets
    cost: Theano variable
        Objective fucntion to minimize

    Notes
    -----
    For more information, see [ADADELTA]_.

    .. [ADADELTA] Matthew D. Zeiler, *ADADELTA: An Adaptive Learning
       Rate Method*, arXiv:1212.5701.
    """

    zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.items()]
    running_up2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                 name='%s_rup2' % k)
                   for k, p in tparams.items()]
    running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.items()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function([x, y], cost, updates=zgup + rg2up,
                                    name='adadelta_f_grad_shared')

    updir = [-tensor.sqrt(ru2 + 1e-6) / tensor.sqrt(rg2 + 1e-6) * zg
             for zg, ru2, rg2 in zip(zipped_grads,
                                     running_up2,
                                     running_grads2)]
    ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2))
             for ru2, ud in zip(running_up2, updir)]
    param_up = [(p, p + ud) for p, ud in zip(tparams.values(), updir)]

    f_update = theano.function([lr], [], updates=ru2up + param_up,
                               on_unused_input='ignore',
                               name='adadelta_f_update')

    return f_grad_shared, f_update

def rmsprop(lr, tparams, grads, x, y, cost, momentum=0):
    """
    A variant of  SGD that scales the step size by running average of the
    recent step norms.

    Parameters
    ----------
    lr : Theano SharedVariable
        Initial learning rate
    tpramas: Theano SharedVariable
        Model parameters
    grads: Theano variable
        Gradients of cost w.r.t to parameres
    x: Theano variable
        Model inputs
    mask: Theano variable
        Sequence mask
    y: Theano variable
        Targets
    cost: Theano variable
        Objective fucntion to minimize

    Notes
    -----
    For more information, see [Hint2014]_.

    .. [Hint2014] Geoff Hinton, *Neural Networks for Machine Learning*,
       lecture 6a,
       http://cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
    """

    zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.items()]
    running_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                   name='%s_rgrad' % k)
                     for k, p in tparams.items()]
    running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.items()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rgup = [(rg, 0.95 * rg + 0.05 * g) for rg, g in zip(running_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function([x, y], cost,
                                    updates=zgup + rgup + rg2up,
                                    name='rmsprop_f_grad_shared')

    updir = [theano.shared(p.get_value() * numpy_floatX(0.),
                           name='%s_updir' % k)
             for k, p in tparams.items()]
    updir_new = [(ud, 0.9 * ud - 1e-4 * zg / tensor.sqrt(rg2 - rg ** 2 + 1e-4))
                 for ud, zg, rg, rg2 in zip(updir, zipped_grads, running_grads,
                                            running_grads2)]
    param_up = [(p, p + udn[1])
                for p, udn in zip(tparams.values(), updir_new)]
    f_update = theano.function([lr], [], updates=updir_new + param_up,
                               on_unused_input='ignore',
                               name='rmsprop_f_update')

    return f_grad_shared, f_update

def train_brnn(
    patience=10,  # Number of epoch to wait before early stop if no progress
    max_epochs=5000,  # The maximum number of epoch to run
    dispFreq=1,  # Display to stdout the training progress every N updates
    lrate=1e-4,  # Learning rate for sgd (not used for adadelta and rmsprop)
    optimizer=rmsprop,  # sgd, adadelta and rmsprop available, sgd very hard to use,
                         # not recommanded (probably need momentum and decaying learning rate).
    saveto='../model/batch_64_pretrain_no_dropout_no_blur_14_x4_2.npz',  # The best model will be saved there
    validFreq=10,  # Compute the validation error after this number of update.
    saveFreq=50,  # Save the parameters after every saveFreq updates
    batch_size=64,  # The batch size during training and validateing.

    # Parameter for extra option
    noise_std=0.,
    momentum = 0,
    reload_model=False,  # Path to a saved model we want to start from.
    use_dropout=False  # if False slightly faster, but worst test error
                      # This frequently need a bigger model.
):
    """
    The main body

    :param patience:
    :param max_epochs:
    :param dispFreq:
    :param lrate:
    :param optimizer:
    :param saveto:
    :param validFreq:
    :param saveFreq:
    :param batch_size:
    :param noise_std:
    :param use_dropout:
    :return:
    """
    options = locals().copy()


    print('... Loading data')
    train_path = '../data/14_seq_41085_yuv_scala_4_frm10_blur_14.mat'
    # train_path = '../data/test/4480_yuv_scala_4_frm10_Dirty_Dancing_blur_14.mat'
    valid_path = OrderedDict()

    valid_path['Dirty_Dancing'] = '../data/test/4480_yuv_scala_4_frm10_Dirty_Dancing_blur_14.mat'
    valid_path['Turbine'] = '../data/test/5590_yuv_scala_4_frm10_Turbine_blur_14.mat'
    valid_path['Star_Fan'] = '../data/test/11100_yuv_scala_4_frm10_Star_Fan_blur_14.mat'
    valid_path['Flag'] = '../data/test/5148_yuv_scala_4_frm10_Flag_blur_14.mat'
    valid_path['Treadmill'] = '../data/test/4070_yuv_scala_4_frm10_Treadmill_blur_14.mat'

    train_set_x, train_set_y = load_data(train_path)
    mean = np.mean(train_set_x)
    print('the mean of train set is: ' + str(mean))
    train_set_x = train_set_x - mean
    train_set_y = train_set_y - mean

    valid_set = OrderedDict()
    for k, v in valid_path.iteritems():
        valid_set_x, valid_set_y = load_data(v)
        valid_set[k] = [valid_set_x - mean, valid_set_y - mean]

    options['size_spa'] = 32
    options['stride_spa'] = 14
    options['stride_tem'] = 8
    options['size_tem'] = 10
    options['scale'] = 4
    options['n_timestep'] = 10
    options['filter_shape'] = [
        [64, 1, 9, 9],
        [32, 64, 1, 1],
        [1, 32, 5, 5]
    ]
    options['rec_filter_size'] = [
        [64, 64, 1, 1],
        [32, 32, 1, 1]
    ]

    options['size'] = [np.concatenate(([batch_size], train_set_x.shape[2:]), axis=0)]

    for i in range(len(options['filter_shape'])):
        options['size'].append([batch_size, options['filter_shape'][i][0], options['size'][-1][-1] - options['filter_shape'][i][-1] + 1,  options['size'][-1][-1] - options['filter_shape'][i][-1] + 1])

    print("model options", options)
    print('... Building model')

    net = BiRecConvNet(options)
    # use_noise is for dropout
    (x, y, f_x, cost, params, use_noise_f, use_noise_b) = net.build_net()

    f_cost = theano.function([x, y], cost, name='f_cost')
    lparams = list(params.values())
    grads = tensor.grad(cost, wrt=lparams)
    f_grad = theano.function([x, y], grads, name='f_grad')

    print('... Optimization')

    lr = tensor.scalar(name='lr')
    f_grad_shared, f_update = optimizer(lr, params, grads, x, y, cost)

    print('... Training')
    print("%d train examples" % ((train_set_x.shape[0] / batch_size) * batch_size))


    kf_valid_set = OrderedDict()
    for k, v in valid_set.iteritems():
        kf_valid = get_minibatches_idx(v[0].shape[0], batch_size, shuffle=True)
        kf_valid_set[k] = kf_valid
        print("vaild set %s with %d examples" % (k, (v[0].shape[0] / batch_size) * batch_size))
    history_errs = []
    best_p = None
    bad_count = 0

    if validFreq == -1:
        validFreq = train_set_x.shape[0] // batch_size
    if saveFreq == -1:
        saveFreq = train_set_x.shape[0] // batch_size

    uidx = 0  # the number of update done
    estop = False  # early stop
    try:
        for eidx in range(max_epochs):
            n_samples = 0

            # Get new shuffled index for the training set.
            kf = get_minibatches_idx(train_set_x.shape[0], batch_size, shuffle=True)

            for train_index in kf[1]:
                uidx += 1
                use_noise_f.set_value(1.)
                use_noise_b.set_value(1.)
                # Select the random examples for this minibatch

                x = np.asarray([train_set_x[t, :, :, :, :] for t in train_index])
                y = np.asarray([train_set_y[t, :, :, 7:27, 7:27] for t in train_index])

                n_samples += x.shape[0]

                x = theano.shared(value=x, borrow=True).dimshuffle(1, 0, 2, 3, 4).eval()
                y = theano.shared(value=y, borrow=True).dimshuffle(1, 0, 2, 3, 4).eval()


                cost = f_grad_shared(x, y)
                f_update(lrate)


                if np.isnan(cost) or np.isinf(cost):
                    print('bad cost detected: ', cost)
                    return 1., 1., 1.

                if np.mod(uidx, dispFreq) == 0:
                    print('Epoch ', eidx, 'Update ', uidx, 'Cost ', cost)

                if saveto and np.mod(uidx, saveFreq) == 0:
                    print('Saving...')

                    if best_p is not None:
                        params = best_p
                    else:
                        params = params
                    np.savez(saveto, history_errs=history_errs, **params)
                    pickle.dump(options, open('%s.pkl' % saveto, 'wb'), -1)
                    print('Done')

                if np.mod(uidx, validFreq) == 0:
                    use_noise_f.set_value(0.)
                    use_noise_b.set_value(0.)

                    valid_psnr = OrderedDict()

                    for k, v in kf_valid_set.iteritems():
                        valid_psnr[k] = pred_error(f_x, valid_set[k][0], valid_set[k][1], v[1], mean)

                    history_errs.append(valid_psnr.values() + [cost])

                    if (best_p is None or
                        cost <= np.array(history_errs)[:, -1].min()):

                        best_p = params
                        bad_counter = 0

                    print('Epoch ', eidx, 'Update ', uidx, ' ...Valid_PSNR: ', valid_psnr)

                    if (len(history_errs) > patience and
                        cost >= np.array(history_errs)[:-patience, -1].min()):
                        bad_counter += 1
                        if bad_counter > patience:
                            lrate /= 10
                            print('Downing learning rate for ', lrate, '\n')
                            bad_counter = 0
                            

    except KeyboardInterrupt:
        print("Training interupted\n")







if __name__ == '__main__':
    train_brnn(max_epochs=10)

