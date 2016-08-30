from convnet3d import ConvLayer 
from convnet3d import PoolLayer 
#from convnet3d import UnPoolLayer 
#from convnet3d import SoftMaxLayer 
#from convnet3d import LogReg3D
from mlp import LogRegr 
from scipy import ndimage
import theano 
from theano import tensor as T 
import numpy as np 
import os 
import sys 
import timeit 
import matplotlib.pyplot as plt
from load_dataset import shared_dataset


filterSize = np.array([[3, 3, 3], [3, 3, 3]])
padSize = np.round(filterSize/2)
''' LOAD DATASET ''' 
trainCTPath = "../Dataset/MICCAI/Train_Resamp/CT/"
testCTPath = "../Dataset/MICCAI/Test_Resamp/CT/"

trainMaskPath = "../Dataset/MICCAI/Train_Resamp/Mask/"
testMaskPath = "../Dataset/MICCAI/Test_Resamp/Mask/"

train_set_x, train_set_y, dtrain_y = shared_dataset(trainCTPath, trainMaskPath, testCTPath, testMaskPath, filterSize, padSize)


''' CREATE MINIBATCHES ''' 
batch_size=2
n_train_batches = train_set_x.get_value(borrow=True).shape[0] 
#n_test_batches = test_set_x.get_value(borrow=True).shape[0] 
n_train_batches /= batch_size 
#n_test_batches /= batch_size 
#n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] 
#n_valid_batches /= batch_size 


''' LAYERS ''' 
rng1 = np.random.RandomState(1234) 
x=T.tensor4('x',dtype=theano.config.floatX) 
y=T.tensor4('y',dtype=theano.config.floatX)

Depth = train_set_x.get_value(borrow=True).shape[1]
Height = train_set_x.get_value(borrow=True).shape[2]
Width = train_set_x.get_value(borrow=True).shape[3]

nkerns=[1, 2, 2] 
learning_rate=0.01
 
input_size = train_set_x.get_value(borrow=True).shape
output_size = train_set_y.shape

layer_0_input=x.reshape((batch_size, 1, input_size[1], input_size[2], input_size[3]))
# Zero padding a tensor



final_output= y.reshape((batch_size, 1, output_size[1], output_size[2], output_size[3]))
# n_in_map = number of input maps (How many maps are there in the input)
# n_out_map = number of output maps (How many maps are there in the output)
#    def __init__(self, input, n_in_maps, n_out_maps, kernel_shape, video_shape, 
#        batch_size, activation, layer_name="Conv", rng=RandomState(1234), 
#        borrow=True, W=None, b=None):
layer0=ConvLayer(layer_0_input, nkerns[0], nkerns[1], (filterSize[0,0], filterSize[0,1], filterSize[0,2]),
                 (input_size[1], input_size[2], input_size[3]), batch_size, T.tanh )


########   Pooling
layer1=PoolLayer(layer0.output, (2,2,2)) 
#T.nnet.abstract_conv.bilinear_upsampling() This function may do upsampling too


########   Depooling
shp_1 = layer1.output.shape
layer2_output = T.zeros((shp_1[0], shp_1[1], shp_1[2]*2, shp_1[3]*2, shp_1[4]*2), dtype=layer1.output.dtype)
layer2_output = T.set_subtensor(layer2_output[:, :, ::2, ::2, ::2], layer1.output)
#

########   Theano Upsampling
zero_padding_1 = T.zeros((batch_size, nkerns[1], input_size[1], input_size[2], input_size[3]), dtype = theano.config.floatX)
layer_3_input = T.set_subtensor(zero_padding_1[:,:,padSize[0,0]:input_size[1]-padSize[0,0], padSize[0,1]:input_size[1]-padSize[0,1],
                                            padSize[0,2]:input_size[2]-padSize[0,2]], layer2_output)

layer3=ConvLayer(layer_3_input, nkerns[1], nkerns[2], (filterSize[1,0], filterSize[1,1], filterSize[1,2]),
                 (input_size[1], input_size[2], input_size[3]), batch_size, T.tanh )


layer3_output = layer3.output

layer4_input_1 = T.squeeze(layer3_output[:,0,:,:,:])
layer4_input_1 = layer4_input_1.reshape([layer4_input_1.shape[0]*layer4_input_1.shape[1]*
                                layer4_input_1.shape[2]*layer4_input_1.shape[3], 1])

layer4_input_2 = T.squeeze(layer3_output[:,1,:,:,:])
layer4_input_2 = layer4_input_2.reshape([layer4_input_2.shape[0]*layer4_input_2.shape[1]*
                                layer4_input_2.shape[2]*layer4_input_2.shape[3], 1])
layer4_input = T.concatenate([layer4_input_1, layer4_input_2], axis=1)

layer4=LogRegr(layer4_input, 2, 1, "relu", rng1)
layer4_output =  layer4.p_y_given_x;
layer4_output = layer4_output.reshape((batch_size, 1, layer3_output.shape[2], layer3_output.shape[3],
                                       layer3_output.shape[4]))
                                       
''' PARAMS, COST, GRAD, UPDATES, ERRORS ''' 
#HiddenLayer

##################   CHANGE ##############
cost = T.mean(T.neq(layer4_output, final_output))
#cost= T.nnet.binary_crossentropy(layer4_output, y).mean()# -T.mean(T.log(L4x)[T.arange(y.shape[0]), y]) 
##################   CHANGE ##############
params=layer0.params + layer3.params + layer4.params
grads=T.grad(cost,params)
#
updates=[(param_i, param_i - learning_rate*grad_i) for param_i,grad_i in zip(params,grads)] 
index=T.iscalar()


' DEFINE TRAIN, TEST & VALIDATION MODEL ''' 
#index is input to the function and layer5.errors is the function output
#layer0_error = T.mean(T.neq(layer0.output, final_output))
##################   CHANGE - CHANGE ##############
#last_layer_error = T.mean(T.neq(layer4_output, final_output))


##################   CHANGE ##############

train_model = theano.function([index], cost, updates=updates,
            givens={ x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]} , on_unused_input='ignore') 


############### 
# TRAIN MODEL # 
############### 
print '... training' 
# early-stopping parameters 
patience = 1000 # look as this many examples regardless 
patience_increase = 2 # wait this much longer when a new best is # found 
improvement_threshold = 0.995 # a relative improvement of this much is considered significant 
validation_frequency = min(n_train_batches, patience / 2) 
# go through this many minibatche before checking the network on the validation set; in this case we check every epoch 
best_validation_loss = np.inf 
best_iter = 0 
test_score = 0. 
start_time = timeit.default_timer() 
epoch = 0 
n_epochs=5000 
done_looping = False 
while (epoch < n_epochs) and (not done_looping): 
    epoch = epoch + 1 
    for minibatch_index in xrange(n_train_batches): 
        iter = (epoch - 1) * n_train_batches + minibatch_index 
        if iter % 100 == 0: 
            print 'training @ iter = ', iter 
            
        cost_ij = train_model(minibatch_index) 
        # for each 10 train sample, validates the model once
#        if (iter + 1) % validation_frequency == 0: 
#            # compute zero-one loss on validation set 
#            validation_losses = [validate_model(i) 
#            for i in xrange(n_valid_batches)] 
#                this_validation_loss = np.mean(validation_losses) 
#            
#            print('epoch %i, minibatch %i/%i, validation error %f %%' % 
#                        (epoch, minibatch_index + 1, n_train_batches, 
#                        this_validation_loss * 100.)) 
#            # if we got the best validation score until now 
#            if this_validation_loss < best_validation_loss: 
#                #improve patience if loss improvement is good enough 
#                if this_validation_loss < best_validation_loss * improvement_threshold: 
#                    patience = max(patience, iter * patience_increase) 
#
#                # save best validation score and iteration number 
#                best_validation_loss = this_validation_loss 
#                best_iter = iter 
#                # test it on the test set 
#                test_losses = [ 
#                    test_model(i) 
#                    for i in xrange(n_test_batches) 
#                ] 
#                test_score = np.mean(test_losses) 
#                print((' epoch %i, minibatch %i/%i, test error of ' 
#                    'best model %f %%') % 
#                    (epoch, minibatch_index + 1, n_train_batches, 
#                     test_score * 100.)) 
        if patience <= iter: 
            done_looping = True 
            break 

end_time = timeit.default_timer() 
print('Optimization complete.') 
#print('Best validation score of %f %% obtained at iteration %i, ' 
#    'with test performance %f %%' % 
#    (best_validation_loss * 100., best_iter + 1, test_score * 100.)) 
print >> sys.stderr, ('The code for file ' + 
                os.path.split(__file__)[1] + 
                ' ran for %.2f seconds' % (end_time - start_time))


#weights = np.squeeze(layer0.W.get_value())
#bias = np.squeeze(layer0.b.get_value())
##################   CHANGE ##############
weights_layer0 = np.squeeze(layer0.W.get_value())
weights_layer3 = np.squeeze(layer3.W.get_value())
weights_layer4 = np.squeeze(layer4.W.get_value())
##################   CHANGE ##############
bias_layer0 = np.squeeze(layer0.b.get_value())
bias_layer3 = np.squeeze(layer3.b.get_value())
bias_layer4 = np.squeeze(layer4.b.get_value())


##################   CHANGE ##############
# Layer_1 result
in_01 = np.squeeze(train_set_x.get_value(borrow=True)[1,1:-1,1:-1,1:-1])
ground_01 = np.squeeze(dtrain_y[1,:,:,:])
#out_01 = np.zeros((nkerns[1], input_size[1], input_size[2], input_size[3]), dtype=np.float32)
out_01 = np.zeros((nkerns[1], input_size[1]-2*padSize[0,0], input_size[2]-2*padSize[0,1],
                         input_size[3]-2*padSize[0,2]), dtype=np.float32)
#out_01_NoPad_Pool = np.zeros((nkerns[1], (input_size[1]-2*padSize[0,0])/2, (input_size[2]-2*padSize[0,1])/2,
#                         (input_size[3]-2*padSize[0,2])/2), dtype=np.float32)
                
for i in range (0,nkerns[1]):
    out_01[i,:,:,:] = ndimage.convolve(in_01, weights_layer0[i,:,:,:], mode='constant', cval=0.0)
    #Padding
#    out_01_NoPad[i,:,:,:] = out_01_Pad[i, 1:-1, 1:-1, 1:-1]
    np.add(out_01[i,:,:,:], bias_layer0[i])
    #Downsampling and Upsampling using scipy functions
    temp1 = ndimage.zoom(out_01[i,:,:,:], .5, order=0) #Pooling
    out_01[i,:,:,:] = ndimage.zoom(temp1[:,:,:], 2, order=0) #Depooling

# Layer_3 result
in_03 = out_01
out_03 = np.zeros((nkerns[2], input_size[1]-2*padSize[1,0], input_size[2]-2*padSize[1,1],
                         input_size[3]-2*padSize[1,2]), dtype=np.float32)

for i in range (0,nkerns[2]):
    out_03[i,:,:,:] = np.sum(ndimage.convolve(in_03, weights_layer3[i,:,:,:], mode='constant', cval=0.0), axis=0)
    np.add(out_03[i,:,:,:], bias_layer3[i])


# Layer_4 result
tShape = out_03.shape
temp1 = np.reshape(out_03[0,:,:,:], (out_03.shape[1]*out_03.shape[2]*out_03.shape[3], 1))
temp2 = np.reshape(out_03[1,:,:,:], (out_03.shape[1]*out_03.shape[2]*out_03.shape[3], 1))
in_04 = np.concatenate((temp1, temp2), axis=1)
out_04 = np.add(np.dot(in_04, weights_layer4), bias_layer4)
out_04 = np.reshape(out_04, (tShape[1], tShape[2], tShape[3]))


sliceNo = 15;
plt.subplot(131)
plt.imshow(np.squeeze(in_01[sliceNo,:,:]), cmap='Greys')
plt.title('Input image')
plt.subplot(132)
plt.imshow(np.squeeze(out_04[sliceNo,:,:]), cmap='Greys')
plt.title('Processed image')
plt.subplot(133)
plt.imshow(np.squeeze(ground_01[sliceNo,:,:]), cmap='Greys')
plt.title('Processed image')
