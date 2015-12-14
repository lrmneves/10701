from sklearn.metrics.pairwise import cosine_similarity,euclidean_distances
from scipy.cluster.vq import *
from sklearn.preprocessing import StandardScaler
import cv2
import numpy as np
import theano
import theano.tensor as T
import lasagne
import cPickle as pickle
X,Y = np.load("X_class.npy"),np.load("Y_class.npy")
#X,Y = np.load("X_car_class.npy"),np.load("Y_car_class.npy")

MAX_LENGTH = 20 


# Number of units in the hidden (recurrent) layer
N_HIDDEN =2048
# Number of training sequences in each batch
N_BATCH = len(X)
# Optimization learning rate
LEARNING_RATE = 1e-4
# All gradients above this will be clipped
GRAD_CLIP = 100
# How often should we check the output?
EPOCH_SIZE = 10
# Number of epochs to train the net
NUM_EPOCHS = 1000

NUM_FEATURES = len(X[0][0])

    	
mask = np.zeros((N_BATCH, MAX_LENGTH))
for b in range(N_BATCH):
    mask[b,:len(X[b])] = 1





# First, we build the network, starting with an input layer
# Recurrent layers expect input of shape
# (batch size, max sequence length, number of features)
l_in = lasagne.layers.InputLayer(shape=(N_BATCH, MAX_LENGTH, NUM_FEATURES))
# l_dropout = lasagne.layers.DropoutLayer(l_in,p=0.9)
l_mask = lasagne.layers.InputLayer(shape=(N_BATCH, MAX_LENGTH))

# The network also needs a way to provide a mask for each sequence.  We'll
# use a separate input layer for that.  Since the mask only determines
# which indices are part of the sequence for each batch entry, they are
# supplied as matrices of dimensionality (N_BATCH, MAX_LENGTH)
# We're using a bidirectional network, which means we will combine two
# RecurrentLayers, one with the backwards=True keyword argument.
# Setting a value for grad_clipping will clip the gradients in the layer
# Setting only_return_final=True makes the layers only return their output
# for the final time step, which is all we need for this task
l_forward = lasagne.layers.LSTMLayer(
    l_in, N_HIDDEN, mask_input=l_mask, grad_clipping=GRAD_CLIP,
    nonlinearity=lasagne.nonlinearities.tanh,learn_init = True, only_return_final=False)

l_backward = lasagne.layers.LSTMLayer(
    l_in, N_HIDDEN, mask_input=l_mask, grad_clipping=GRAD_CLIP,
    nonlinearity=lasagne.nonlinearities.tanh,
    only_return_final=False,learn_init = True)

l_concat = lasagne.layers.ConcatLayer([l_forward, lasagne.layers.dropout(l_backward, p=0.5)])

l_forward_2 = lasagne.layers.LSTMLayer(
        lasagne.layers.dropout(l_concat, p=0.5), N_HIDDEN, grad_clipping=GRAD_CLIP,
        nonlinearity=lasagne.nonlinearities.softplus,learn_init=True)

# l_forward_3 = lasagne.layers.LSTMLayer(
#         lasagne.layers.dropout(l_forward_2, p=0.2), N_HIDDEN, grad_clipping=GRAD_CLIP,
#         nonlinearity=lasagne.nonlinearities.leaky_rectify,learn_init=True)
# l_backward_2 = lasagne.layers.LSTMLayer(
#         lasagne.layers.dropout(l_forward_1, p=.2), N_HIDDEN, grad_clipping=GRAD_CLIP,
#         nonlinearity=lasagne.nonlinearities.leaky_rectify,learn_init=True,backwards = True)

# l_concat = lasagne.layers.ConcatLayer([l_forward_2, l_backward_2])
# l_forward_slice = lasagne.layers.SliceLayer(l_forward_3, indices=slice(0,FINAL_FEATURES))

l_out = lasagne.layers.DenseLayer(l_forward_2, num_units=1, W = lasagne.init.GlorotNormal(), 
    nonlinearity=lasagne.nonlinearities.sigmoid)

target_values = T.vector('target_output')
# lasagne.layers.get_output produces a variable for the output of the net

network_output = lasagne.layers.get_output(l_out,deterministic = False)
predicted_values = network_output.flatten()

# predictions = network_output*255

# l1_reg = lasagne.regularization.regularize_layer_params(l_in,lasagne.regularization.l1)*1e-2

# l2_reg = lasagne.regularization.regularize_layer_params(l_forward_slice,lasagne.regularization.l2)*1e-4

cost = lasagne.objectives.aggregate(lasagne.objectives.binary_crossentropy(predicted_values,target_values), mode='mean')
# print cost
# Retrieve all parameters from the network
all_params = lasagne.layers.get_all_params(l_out)
# Compute SGD updates for training
# print("Computing updates ...")
updates = lasagne.updates.adagrad(cost, all_params, LEARNING_RATE)
# Theano functions for training and computing cost
# print("Compiling functions ...")

train = theano.function([l_in.input_var, target_values,l_mask.input_var], 
    cost, 
    updates=updates, 
    allow_input_downcast=True)



compute_cost = theano.function([l_in.input_var, target_values, l_mask.input_var], cost,allow_input_downcast=True)
get_pred = theano.function([l_in.input_var,l_mask.input_var],lasagne.layers.get_output(l_out,deterministic = True),allow_input_downcast=True)

probs = theano.function([l_in.input_var,l_mask.input_var],network_output,allow_input_downcast=True)
# pred_func = theano.function([l_in.input_var],lasagne.layers.get_output(l_out,deterministic = True),allow_input_downcast=True)
print("Training ...")
try:
    for epoch in range(NUM_EPOCHS):
        for _ in range(EPOCH_SIZE):
            
            train(np.array(X), np.array(Y),mask)

        cost_val = compute_cost(np.array(X), np.array(Y),mask)
        print("Epoch {} validation cost = {}".format(epoch, cost_val))
        
        if cost_val < 1e-3:
            break;

except KeyboardInterrupt:
    pass
#path = "/home/public/10701/feature/car_feature_414.npy"
path = "/home/public/10701/feature/mnist_feature_50_20_alex.npy"
X_full = np.load(path)
#X_full = X_full[20:40]
start_size = 10
#X_t = X_full
X_t = X_full[1]
sequence = X_t[:start_size]
idx_sequence = []
frames_left = set()
for i in range(start_size, len(X_t)):
	frames_left.add(i)
training_size = 100

while len(frames_left) > 0:
	probability = []
        z_l = np.zeros((training_size-1,MAX_LENGTH,len(X_t[0])))
	for f in frames_left:
		current = np.vstack([sequence,X_t[f]])
		mask_p = np.zeros((training_size,MAX_LENGTH))
		mask_p[0,:len(current)] = 1
		z = np.zeros((MAX_LENGTH-len(current),len(X_t[0])))
		current = np.vstack([np.array(current),z])
		current = np.vstack([[current],z_l])
		next_frame = get_pred(current,mask_p)[0]
		probability.append((next_frame,f))
	
        probability= sorted(probability)
	print probability
        idx_sequence.append(probability[0][1])
	frames_left.remove(probability[0][1])



print idx_sequence

pickle.dump( lasagne.layers.get_all_param_values(l_out), open( "params.p", "wb" ) )
