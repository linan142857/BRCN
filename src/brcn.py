from __future__ import print_function
from PIL import Image
import h5py
import numpy as np
import theano
import theano.tensor as tensor
from theano import config
from theano.tensor.nnet.conv import conv2d
from collections import OrderedDict


"""
a random number generator used to initialize weights
"""
SEED = 123
rng = np.random.RandomState(SEED)

class ConvLayer(object):
    """
    Pool Layer of a convolutional network
    """

    def __init__(self, filter_shape):
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
        self.W = theano.shared(
                np.asarray(
                    rng.normal(0, 1e-3, size=filter_shape),
                    dtype=config.floatX
                ),
                borrow=True
            )

        # the bias is a 1D tensor -- one bias per output feature map
        self.b = theano.shared(np.zeros(filter_shape[0]).astype(config.floatX), borrow=True)

        # store parameters of this layer
        self.params = [self.W, self.b]

    def conv(self, input):
        # convolve input feature maps with filters
        conv_out = conv2d(
            input=input,
            filters=self.W,
            border_mode='valid'
        )
        output = conv_out + self.b.dimshuffle('x', 0, 'x', 'x')

        return output

class BiRecConvNet(object):
    """Bidirectional Recurrent Convolutional Networks"""

    def __init__(self, options):
        self.options = options

    def build_net(self, model):
        options = self.options
        # forward & backward data flow
        x = tensor.TensorType(dtype=config.floatX, broadcastable=(False,) * 5)(name='x')
        y = tensor.TensorType(dtype=config.floatX, broadcastable=(False,) * 5)(name='y')

        # forward net
        layers_f, params_f = self._init_layer(options['filter_shape'], options['rec_filter_size'])
        proj_f = self._build_model(x, options, layers_f, params_f, go_backwards=False)

        # backward net
        layers_b, params_b = self._init_layer(options['filter_shape'], options['rec_filter_size'])
        proj_b = self._build_model(x, options, layers_b, params_b, go_backwards=True)

        params = dict(prefix_p('f', params_f), **(prefix_p('b', params_b)))

        if model is not None:
            for k in params.iterkeys():
                params[k].set_value(model[k])

        proj = proj_f + proj_b[::-1]

        weight_decay = theano.shared(numpy_floatX(0.), borrow=True)

        for v in params.itervalues():
            weight_decay += (v ** 2).sum()

        cost = ((y - proj) ** 2).sum() / (x.shape[1] * x.shape[0]) + 1e-4 * weight_decay
        f_x = theano.function([x], proj, name='f_proj')

        return x, y, f_x, cost, params

    def _init_layer(self, filter_shape, rec_filter_size):
        """
        Global (net) parameter. For the convolution and regular opt.
        """
        layers = OrderedDict()
        params = OrderedDict()


        for i in range(len(filter_shape)):
            layers['conv_' + str(i) + '_v'] = ConvLayer(filter_shape[i])
            layers['conv_' + str(i) + '_t'] = ConvLayer(filter_shape[i])
            params['conv_' + str(i) + '_v_w'] = layers['conv_' + str(i) + '_v'].params[0]
            params['conv_' + str(i) + '_v_b'] = layers['conv_' + str(i) + '_v'].params[1]
            params['conv_' + str(i) + '_t_w'] = layers['conv_' + str(i) + '_t'].params[0]
            params['conv_' + str(i) + '_t_b'] = layers['conv_' + str(i) + '_t'].params[1]

            if i < len(rec_filter_size):
                layers['conv_' + str(i) + '_r'] = ConvLayer(rec_filter_size[i])
                params['conv_' + str(i) + '_r_w'] = layers['conv_' + str(i) + '_r'].params[0]
                params['conv_' + str(i) + '_r_b'] = layers['conv_' + str(i) + '_r'].params[1]

            params['b_' + str(i)] = theano.shared(np.zeros(filter_shape[i][0]).astype(config.floatX), name='b_' + str(i), borrow=True)


        return layers, params

    def _build_model(self, input, options, layers, params, go_backwards=False):

        def _step1(x_, t_, layer_):
            layer_ = str(layer_.data)
            v = layers['conv_' + layer_ + '_v'].conv(x_)
            t = layers['conv_' + layer_ + '_t'].conv(t_)
            h = v + t

            return x_, h

        def _step2(h, r_, layer_):
            layer_ = str(layer_.data)
            o = h + params['b_' + layer_].dimshuffle('x', 0, 'x', 'x')
            if layer_ != str(len(options['filter_shape']) - 1):
                r = layers['conv_' + layer_ + '_r'].conv(r_)
                o = tensor.nnet.relu(o + r)
            return o

        rval = input
        if go_backwards:
            rval = rval[::-1]
        for i in range(len(options['filter_shape'])):
            rval, _ = theano.scan(_step1, sequences=[rval],
                                  outputs_info=[rval[0], None],
                                  non_sequences=[i],
                                  name='rnn_layers_k_' + str(i))
            rval = rval[1]
            rval, _ = theano.scan(_step2, sequences=[rval],
                                  outputs_info=[rval[-1]],
                                  non_sequences=[i],
                                  name='rnn_layers_q_' + str(i))
        proj = rval

        return proj

def pred_error(f_pred, data, target, options, uidx, k):
    """
    Just compute the error
    f_pred: Theano fct computing the prediction
    target: usual groundtruth for that dataset.
    """
    cut = 30
    diff = options['padding']
    x = data
    y = target[:, :, :, diff:-diff, diff:-diff]
    x = theano.shared(x).dimshuffle(1, 0, 2, 3, 4).eval()
    y = theano.shared(y).dimshuffle(1, 0, 2, 3, 4).eval()
    numfrm = x.shape[0]
    psnr_err = 0.
    for i in range(0, numfrm, cut):
        _x = x[i:i+cut]
        _y = y[i:i+cut]
        pred = f_pred(_x)
        pred = np.around(pred * 255)
        _y = np.around(_y * 255)

        psnr_err += pnsr(pred, _y)
        if i is cut:
            img_x = pred[0, 0, 0, :, :].astype(np.uint8).transpose()
            img_y = _y[0, 0, 0, :, :].astype(np.uint8).transpose()
            img_b = (_x[0, 0, 0, diff:-diff, diff:-diff] * 255).astype(np.uint8).transpose()

    psnr_err /= numfrm

    im = Image.frombytes('L', img_b.shape, img_b)
    im.save('../photo/' + k + '_bic.png', "PNG")
    im = Image.frombytes('L', img_y.shape, img_y)
    im.save('../photo/' + k + '_gth.png', "PNG")
    im = Image.frombytes('L', img_x.shape, img_x)
    im.save('../photo/' + k + '_' + str(uidx) + '_' + str(psnr_err) + '_prd.png', "PNG")


    return psnr_err

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
    npy = np.load(path)
    return npy.all()

def pnsr(x, y):
    z = np.sum((y - x) ** 2, axis=(2, 3, 4))
    z /= x.shape[2] * x.shape[3] * x.shape[4]
    rmse = np.sqrt(z)
    psnr = 20 * np.log10(255 / rmse)
    psnr = np.sum(psnr)
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
    max_epochs=1000,  # The maximum number of epoch to run
    dispFreq=1,  # Display to stdout the training progress every N updates
    lrate=1e-4,  # Learning rate for sgd (not used for adadelta and rmsprop)
    optimizer=rmsprop,  # sgd, adadelta and rmsprop available, sgd very hard to use,
                         # not recommanded (probably need momentum and decaying learning rate).
    saveto='14_seq_41085_yuv_scala_4_frm10_blur_2',  # The best model will be saved there
    model_path='14_seq_41085_yuv_scala_4_frm10_blur_2.npy',  # The model path
    validFreq=50,  # Compute the validation error after this number of update.
    saveFreq=200,  # Save the parameters after every saveFreq updates
    batch_size=64,  # The batch size during training and validateing.
    # Parameter for extra option
    momentum = 0,
    lmodel=True,  # Path to a saved model we want to start from.
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
    :param lmodel:
    :return:
    """
    options = locals().copy()


    print('... Loading data')
    train_path = '../data/14_seq_41085_yuv_scala_4_frm10_blur_2.mat'
    valid_path = OrderedDict()

    valid_path['Dirty_Dancing'] = '../data/test/58_Dirty_Dancing_scale4_blur_2.mat'
    valid_path['Turbine'] = '../data/test/350_Turbine_scale4_blur_2.mat'
    valid_path['Star_Fan'] = '../data/test/300_Star_Fan_scale4_blur_2.mat'
    valid_path['Flag'] = '../data/test/290_Flag_scale4_blur_2.mat'
    valid_path['Treadmill'] = '../data/test/300_Treadmill_scale4_blur_2.mat'

    train_set_x, train_set_y = load_data(train_path)

    valid_set = OrderedDict()
    for k, v in valid_path.iteritems():
        valid_set_x, valid_set_y = load_data(v)
        valid_set[k] = [valid_set_x, valid_set_y]


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

    options['padding'] = np.sum([(i[-1] - 1) / 2 for i in options['filter_shape']])
    print("model options", options)
    print('... Building model')

    net = BiRecConvNet(options)

    model = None
    if lmodel:
        model = load_model('../model/' + model_path)
    (x, y, f_x, cost, params) = net.build_net(model)

    f_cost = theano.function([x, y], cost, name='f_cost')
    grads = tensor.grad(cost, wrt=list(params.values()))
    f_grad = theano.function([x, y], grads, name='f_grad')

    print('... Optimization')

    lr = tensor.scalar(name='lr')
    f_grad_shared, f_update = optimizer(lr, params, grads, x, y, cost)

    print('... Training')
    print("%d train examples" % ((train_set_x.shape[0] / batch_size) * batch_size))

    history_errs = []
    best_p = None
    bad_count = 0

    if validFreq == -1:
        validFreq = train_set_x.shape[0] // batch_size
    if saveFreq == -1:
        saveFreq = train_set_x.shape[0] // batch_size

    uidx = 0  # the number of update done

    try:
        for eidx in range(max_epochs):

            # Get new shuffled index for the training set.
            kf = get_minibatches_idx(train_set_x.shape[0], batch_size, shuffle=True)

            for train_index in kf[1]:
                uidx += 1
                # Select the random examples for this minibatch
                diff = options['padding']
                x = np.asarray([train_set_x[t, :, :, :, :] for t in train_index])
                y = np.asarray([train_set_y[t, :, :, diff:-diff, diff:-diff] for t in train_index])

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
                    p = dict()
                    for k in params.iterkeys():
                        p[k] = np.asarray(params[k].eval()).astype(config.floatX)
                    np.save('../model/' + saveto, p)
                    print('Done')

                if np.mod(uidx, validFreq) == 0:

                    valid_psnr = OrderedDict()

                    for k, v in valid_set.iteritems():
                        valid_psnr[k] = pred_error(f_x, v[0], v[1], options, uidx, k)

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
                            # lrate /= 10
                            # print('Downing learning rate for ', lrate, '\n')
                            bad_counter = 0


    except KeyboardInterrupt:
        print("Training interupted\n")


if __name__ == '__main__':
    train_brnn(max_epochs=30)
