"""
3D ConvNet layers using Theano, Pylearn and Numpy

ConvLayer: convolutions, filter bank
NormLayer: normalization (LCN, GCN, local mean subtraction)
PoolLayer: pooling, subsampling
RectLayer: rectification (absolute value)

"""


from theano.tensor.nnet.conv3d2d import conv3d
from maxpool3d import max_pool_3d
from activations import relu, softplus

from numpy import sqrt, prod, ones, floor, repeat, pi, exp, zeros, sum
from numpy.random import RandomState

from theano.tensor.nnet import conv2d
from theano import shared, config, _asarray
import theano.tensor as T
floatX = config.floatX


class ConvLayer(object):
    """ Convolutional layer, Filter Bank Layer """

    def __init__(self, input, n_in_maps, n_out_maps, kernel_shape, video_shape, 
        batch_size, activation, layer_name="Conv", rng=RandomState(1234), 
        borrow=True, W=None, b=None):

        """
        video_shape: (frames, height, width)
        kernel_shape: (frames, height, width)

        W_shape: (out, in, kern_frames, kern_height, kern_width)
        """

        self.__dict__.update(locals())
        del self.self
        
        # init W
        if W != None: W_val = W
        else: 
            # fan in: filter time x filter height x filter width x input maps
            fan_in = prod(kernel_shape)*n_in_maps
            norm_scale = 2. * sqrt( 1. / fan_in )
            if activation in (relu,softplus): norm_scale = 0.01
            W_shape = (n_out_maps, n_in_maps)+kernel_shape
            W_val = _asarray(rng.normal(loc=0, scale=norm_scale, size=W_shape),\
                        dtype=floatX)
        self.W = shared(value=W_val, borrow=borrow, name=layer_name+'_W')
        self.params = [self.W]

        # init bias
        if b != None: 
            b_val = b
        elif activation in (relu,softplus): 
            b_val = ones((n_out_maps,), dtype=floatX)
        else: 
            b_val = zeros((n_out_maps,), dtype=floatX)
        self.b = shared(b_val, name=layer_name+"_b", borrow=borrow)
        self.params.append(self.b)

        # 3D convolution; dimshuffle: last 3 dimensions must be (in, h, w)
        n_fr, h, w = video_shape
        n_fr_k, h_k, w_k = kernel_shape
#         if border_mode='valid' it means no padding
        out = conv3d(
                signals=input.dimshuffle([0,2,1,3,4]), 
                filters=self.W, 
                signals_shape=(batch_size, n_fr, n_in_maps, h, w), 
                filters_shape=(n_out_maps, n_fr_k, n_in_maps, h_k, w_k),         
                border_mode='valid').dimshuffle([0,2,1,3,4])
        # if border_mode='full' it means  padding is used so that input and output have the same size
#        out = conv3d(
#                signals=input.dimshuffle([0,2,1,3,4]), 
#                filters=self.W, 
#                signals_shape=(batch_size, n_fr, n_in_maps, h, w), 
#                filters_shape=(n_out_maps, n_fr_k, n_in_maps, h_k, w_k),         
#                border_mode='full').dimshuffle([0,2,1,3,4])
        out += self.b.dimshuffle('x',0,'x','x','x')

        self.output = activation(out)
        

class NormLayer(object):
    """ Normalization layer """

    def __init__(self, input, method="lcn", **kwargs):
        """
        method: "lcn", "gcn", "mean"

        LCN: local contrast normalization
            kwargs: 
                kernel_size=9, threshold=1e-4, use_divisor=True

        GCN: global contrast normalization
            kwargs:
                scale=1., subtract_mean=True, use_std=False, sqrt_bias=0., 
                min_divisor=1e-8

        MEAN: local mean subtraction
            kwargs:
                kernel_size=5
        """

        input_shape = input.shape

        # make 4D tensor out of 5D tensor -> (n_images, 1, height, width)
        input_shape_4D = (input_shape[0]*input_shape[1]*input_shape[2], 1,
                            input_shape[3], input_shape[4])
        input_4D = input.reshape(input_shape_4D, ndim=4)
        if method=="lcn":
            out = self.lecun_lcn(input_4D, **kwargs)
        elif method=="gcn":
            out = self.global_contrast_normalize(input_4D,**kwargs)
        elif method=="mean":
            out = self.local_mean_subtraction(input_4D, **kwargs)
        else:
            raise NotImplementedError()

        self.output = out.reshape(input_shape)

    def lecun_lcn(self, X, kernel_size=7, threshold = 1e-4, use_divisor=False):
        """
        Yann LeCun's local contrast normalization
        Orginal code in Theano by: Guillaume Desjardins
        """

        filter_shape = (1, 1, kernel_size, kernel_size)
        filters = gaussian_filter(kernel_size).reshape(filter_shape)
        filters = shared(_asarray(filters, dtype=floatX), borrow=True)

        convout = conv2d(X, filters=filters, filter_shape=filter_shape, 
                            border_mode='full')

        # For each pixel, remove mean of kernel_sizexkernel_size neighborhood
        mid = int(floor(kernel_size/2.))
        new_X = X - convout[:,:,mid:-mid,mid:-mid]

        if use_divisor:
            # Scale down norm of kernel_sizexkernel_size patch
            sum_sqr_XX = conv2d(T.sqr(T.abs_(new_X)), filters=filters, 
                                filter_shape=filter_shape, border_mode='full')

            denom = T.sqrt(sum_sqr_XX[:,:,mid:-mid,mid:-mid])
            per_img_mean = denom.mean(axis=[2,3])
            divisor = T.largest(per_img_mean.dimshuffle(0,1,'x','x'), denom)
            divisor = T.maximum(divisor, threshold)

            new_X /= divisor

        return new_X#T.cast(new_X, floatX)

    def local_mean_subtraction(self, X, kernel_size=5):
         
        filter_shape = (1, 1, kernel_size, kernel_size)
        filters = mean_filter(kernel_size).reshape(filter_shape)
        filters = shared(_asarray(filters, dtype=floatX), borrow=True)

        mean = conv2d(X, filters=filters, filter_shape=filter_shape, 
                        border_mode='full')
        mid = int(floor(kernel_size/2.))

        return X - mean[:,:,mid:-mid,mid:-mid]

    def global_contrast_normalize(self, X, scale=1., subtract_mean=True, 
        use_std=False, sqrt_bias=0., min_divisor=1e-8):

        ndim = X.ndim
        if not ndim in [3,4]: raise NotImplementedError("X.dim>4 or X.ndim<3")

        scale = float(scale)
        mean = X.mean(axis=ndim-1)
        new_X = X.copy()

        if subtract_mean:
            if ndim==3:
                new_X = X - mean[:,:,None]
            else: new_X = X - mean[:,:,:,None]

        if use_std:
            normalizers = T.sqrt(sqrt_bias + X.var(axis=ndim-1)) / scale
        else:
            normalizers = T.sqrt(sqrt_bias + (new_X ** 2).sum(axis=ndim-1)) / scale

        # Don't normalize by anything too small.
        T.set_subtensor(normalizers[(normalizers < min_divisor).nonzero()], 1.)

        if ndim==3: new_X /= normalizers[:,:,None]
        else: new_X /= normalizers[:,:,:,None]

        return new_X


class PoolLayer(object):
    """ Subsampling and pooling layer """

    def __init__(self, input, pool_shape, method="max"):
        """
        method: "max", "avg", "L2", "L4", ...
        """

        self.__dict__.update(locals())
        del self.self

        if method=="max":
            out = max_pool_3d(input,pool_shape)
        else:
            raise NotImplementedError()

        self.output = out




class UnPoolLayer(object):
    """ Subsampling and pooling layer """

    def __init__(self, input):
        """
        method: "max", "avg", "L2", "L4", ...
        """

#        self.__dict__.update(locals())
#        del self.self
        
        out = T.extra_ops.repeat(input, 2, axis=1)
        out = T.extra_ops.repeat(out, 2, axis=2)
        out = T.extra_ops.repeat(out, 2, axis=3)

        self.output = out
###########################################################################################################
class LogReg3D(object):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, input, n_in, n_out):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        """
        # start-snippet-1
        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.W = theano.shared(
            value=numpy.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )
        # initialize the biases b as a vector of n_out 0s
        self.b = theano.shared(
            value=numpy.zeros(
                (n_out,),
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )

        # symbolic expression for computing the matrix of class-membership
        # probabilities
        # Where:
        # W is a matrix where column-k represent the separation hyperplane for
        # class-k
        # x is a matrix where row-j  represents input training sample-j
        # b is a vector where element-k represent the free parameter of
        # hyperplane-k
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        # symbolic description of how to compute prediction as class whose
        # probability is maximal
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        # end-snippet-1

        # parameters of the model
        self.params = [self.W, self.b]

        # keep track of model input
        self.input = input

    def negative_log_likelihood(self, y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        .. math::

            \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
            \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|}
                \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
            \ell (\theta=\{W,b\}, \mathcal{D})

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label

        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """
        # start-snippet-2
        # y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of examples (call it n) in the minibatch
        # T.arange(y.shape[0]) is a symbolic vector which will contain
        # [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
        # Log-Probabilities (call it LP) with one row per example and
        # one column per class LP[T.arange(y.shape[0]),y] is a vector
        # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
        # LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
        # the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
        # end-snippet-2

    def errors(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()


###########################################################################################################
class SoftMaxLayer(object):
    """ Subsampling and pooling layer """

    def __init__(self, input):
        """
        method: "max", "avg", "L2", "L4", ...
        """

#        self.__dict__.update(locals())
#        del self.self
        
        out = T.nnet.nnet.softmax(input)

        self.output = out


class RectLayer(object):
    """  Rectification layer """

    def __init__(self, input):
        self.output = T.abs_(input)


def gaussian_filter(kernel_shape):

    x = zeros((kernel_shape, kernel_shape), dtype='float32')

    def gauss(x, y, sigma=2.0):
        Z = 2 * pi * sigma**2
        return  1./Z * exp(-(x**2 + y**2) / (2. * sigma**2))

    mid = floor(kernel_shape/ 2.)
    for i in xrange(0,kernel_shape):
        for j in xrange(0,kernel_shape):
            x[i,j] = gauss(i-mid, j-mid)

    return x / sum(x)


def mean_filter(kernel_size):
    s = kernel_size**2
    x = repeat(1./s, s).reshape((kernel_size, kernel_size))
    return x
