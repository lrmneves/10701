
import os
os.sys.path.insert(1, '/home/public/.local/lib/python2.7/site-packages/')

from sklearn.metrics.pairwise import cosine_similarity,euclidean_distances
from scipy.cluster.vq import *
from sklearn.preprocessing import StandardScaler
import cv2
import numpy as np
import theano
import theano.tensor as T
import lasagne

def tnorm(tens):
    '''
    Tensor Norm
    '''
    return T.sqrt(T.sum(T.sqr(tens)))

def mse(x, t):
    """Calculates the MSE mean across all dimensions, i.e. feature
     dimension AND minibatch dimension.
    :parameters:
        - x : predicted values
        - t : target values
    :returns:
        - output : the mean square error across all dimensions
    """
    return (x - t) ** 2
# dataset = np.load("new_data.npy")
# new_data = []
# for frame in dataset:
# 	new_frames = []
# 	for im in range(len(frame)):
# 		new_frames.append(frame[im].flatten())
# 	new_data.append(new_frames)

X_full = np.load("/home/public/10701/rnn/mnist_feature_100_20.npy")
X_t = X_full[0]

X2= X_full[0][:-1]

# Y1= X_full[1][1:]
X1,Y = np.load("X.npy"),np.load("Y.npy")
X = np.hstack([X1,X2])
# Y = np.hstack([Y1,Y2])
FINAL_FEATURES = X1.shape[1]


# Min/max sequence length
MIN_LENGTH = 10
MAX_LENGTH = 3
# Number of units in the hidden (recurrent) layer
N_HIDDEN = 512
# Number of training sequences in each batch
N_BATCH = 1
# Optimization learning rate
LEARNING_RATE = 1e-3
# All gradients above this will be clipped
GRAD_CLIP = 100
# How often should we check the output?
EPOCH_SIZE = 10
# Number of epochs to train the net
NUM_EPOCHS = 1000

NUM_FEATURES = len(X[0])
# X = [f[0] for f in dataset]
# Y = X[1:]
# X = X[:-1]
#np.save("X.npy",np.array(X))
# np.save("Y.npy",np.array(Y,dtype = "int32"))

# X1 = X[:5]
# X2 = X[5:10]
# X3 = X[10:15]
# Y1 = X1[1:]
# Y2 = X2[1:]
# Y3 = X3[1:]
# X1 = X1[:-1]
# X2 = X2[:-1]
# X3 = X3[:-1]

# X = []
# Y = []
# for j in range(4):
# 	current = [[X1[j],X2[j],X3[j]]]
# 	ycurr = [[Y1[j],Y2[j],Y3[j]]]
# 	if X == []:
# 		X = current
# 		Y = ycurr
# 	else:
# 		X = np.vstack([X,current])
# 		Y = np.vstack([Y,ycurr])

# Y = np.array(Y,dtype = "int32")

# print Y.shape
# print X.shape

# current = np.array([f[i] for f in dataset])
# 	X = np.vstack([X,current])
# X1 = np.array([f[25] for f in dataset])
# y = np.array([i for i in range(len(X))],dtype = "int32")
# size = 10

X_train = []
Y_train = []
X_test = []
Y_test = []

for b in range(N_BATCH):
    X_train.append([])

    X_test.append([])
    # Y_train.append(Y[b*MAX_LENGTH + MAX_LENGTH-1])
    # Y_test.append(Y[b*MAX_LENGTH + MAX_LENGTH +9-1])
    for l in range(MAX_LENGTH-1):

        current = X[b*(MAX_LENGTH) + l]
        current 
    	

    	
    	try:
    		X_test[b].append(X[b*MAX_LENGTH + l+MAX_LENGTH*N_BATCH])
    		
    	except Exception:
    		continue
    Y_train.append(Y[b*MAX_LENGTH + MAX_LENGTH-1])
    try:
        Y_test.append(Y[b*MAX_LENGTH + MAX_LENGTH-1+MAX_LENGTH*N_BATCH])
    except Exception: 
        continue 
    X_train[b].append()
    X_train[b] = X_train[b][::1]
X_test = np.array([X[11::1]])
Y_test = np.array([Y[-1]])
# mask_train = np.zeros((N_BATCH, MAX_LENGTH))
# mask_test = np.zeros((N_BATCH, MAX_LENGTH))
# for b in range(N_BATCH):
# 	mask_train[b,:len(X_train[b])] = 1
# 	mask_test[b,:len(X_test[b])] = 1
X_train = np.array(X_train)
Y_train = np.array(Y_train)

# print X_train.shape

X_test= np.array(X_test)
Y_test = np.array(Y_test)
print X_train.shape
print Y_train.shape

# First, we build the network, starting with an input layer
# Recurrent layers expect input of shape
# (batch size, max sequence length, number of features)
l_in = lasagne.layers.InputLayer(shape=(None, MAX_LENGTH, NUM_FEATURES))
# l_dropout = lasagne.layers.DropoutLayer(l_in,p=0.9)

# The network also needs a way to provide a mask for each sequence.  We'll
# use a separate input layer for that.  Since the mask only determines
# which indices are part of the sequence for each batch entry, they are
# supplied as matrices of dimensionality (N_BATCH, MAX_LENGTH)
# We're using a bidirectional network, which means we will combine two
# RecurrentLayers, one with the backwards=True keyword argument.
# Setting a value for grad_clipping will clip the gradients in the layer
# Setting only_return_final=True makes the layers only return their output
# for the final time step, which is all we need for this task
l_forward_1 = lasagne.layers.LSTMLayer(
    l_in, N_HIDDEN, grad_clipping=GRAD_CLIP,
    nonlinearity=lasagne.nonlinearities.tanh, learn_init=True,)

l_forward_2 = lasagne.layers.LSTMLayer(
        lasagne.layers.dropout(l_forward_1, p=0.9), N_HIDDEN, grad_clipping=GRAD_CLIP,
        nonlinearity=lasagne.nonlinearities.sigmoid,learn_init=True)

l_forward_3 = lasagne.layers.LSTMLayer(
        lasagne.layers.dropout(l_forward_2, p=0.2), N_HIDDEN, grad_clipping=GRAD_CLIP,
        nonlinearity=lasagne.nonlinearities.leaky_rectify,learn_init=True)
# l_backward_2 = lasagne.layers.LSTMLayer(
#         lasagne.layers.dropout(l_forward_1, p=.2), N_HIDDEN, grad_clipping=GRAD_CLIP,
#         nonlinearity=lasagne.nonlinearities.leaky_rectify,learn_init=True,backwards = True)

# l_concat = lasagne.layers.ConcatLayer([l_forward_2, l_backward_2])
l_forward_slice = lasagne.layers.SliceLayer(l_forward_3, indices=slice(0,FINAL_FEATURES))

l_out = lasagne.layers.DenseLayer(l_forward_slice, num_units=FINAL_FEATURES, W = lasagne.init.GlorotNormal(), 
    nonlinearity=lasagne.nonlinearities.linear)

target_values = T.matrix('target_output')

# lasagne.layers.get_output produces a variable for the output of the net

network_output = lasagne.layers.get_output(l_out,deterministic = False)
predictions = network_output*255

l1_reg = lasagne.regularization.regularize_layer_params(l_in,lasagne.regularization.l1)*1e-2

l2_reg = lasagne.regularization.regularize_layer_params(l_forward_slice,lasagne.regularization.l2)*1e-4

# cost = lasagne.objectives.squared_error(predictions, target_values).mean() + l1_reg - l2_reg  #mse
cost = T.mean(1 - T.batched_dot(predictions,target_values)/(tnorm(predictions)*tnorm(target_values)))#cosine similarity
# cost = T.mean(T.sum(abs(predictions - target_values))) + l2_reg + l1_reg#manhattan distance
# cost = T.mean(tnorm(predictions - target_values))+ l1_reg #+ l2_reg + l1_reg#euclidean distance
# print cost
# Retrieve all parameters from the network
all_params = lasagne.layers.get_all_params(l_out)
# Compute SGD updates for training
# print("Computing updates ...")
updates = lasagne.updates.adagrad(cost, all_params, LEARNING_RATE)
# Theano functions for training and computing cost
# print("Compiling functions ...")
train = theano.function([l_in.input_var, target_values], cost, updates=updates, allow_input_downcast=True)
compute_cost = theano.function([l_in.input_var, target_values], cost, allow_input_downcast=True)

probs = theano.function([l_in.input_var],network_output,allow_input_downcast=True)
pred_func = theano.function([l_in.input_var],lasagne.layers.get_output(l_out,deterministic = True),allow_input_downcast=True)


print("Training ...")
try:
    for epoch in range(NUM_EPOCHS):
        for _ in range(EPOCH_SIZE):
            
            train(X_train, Y_train)

        cost_val = compute_cost(np.array(X_train), np.array(Y_train))
        print("Epoch {} validation cost = {}".format(epoch, cost_val))

        # pred = pred_func(np.array([X_train[-1]]))
        # pred = np.array(255*pred,dtype = "uint8")
        # pred = pred.reshape(64,64)

        # cv2.imshow("pred",Y_train[-1].reshape(64,64).astype("uint8"))
        # cv2.waitKey(500)
        # cv2.imshow("pred",pred)
        # cv2.waitKey(1000)
        count = 0.0

        start_size = 10

        sequence = [np.array(X[i]) for i in range(start_size)]
        idx_sequence = [range(start_size)]
        frames_left = set()
        next_frame = pred_func(np.array([sequence]))
        cv2.imshow("pred",(255*(next_frame)).reshape(64,64).astype("uint8"))
        cv2.waitKey(1000)
        for i in range(start_size, len(X)):
            frames_left.add(i)

        while len(frames_left) > 0:
            print np.array([sequence]).shape
            next_frame = pred_func(np.array([sequence]))

            next_frame = np.array(255*next_frame,dtype = "uint8")
            # cv2.imshow("pred",(next_frame).reshape(64,64).astype("uint8"))
            # cv2.waitKey(1000)
            dist = [(euclidean_distances(next_frame,Y[j]),j) for j in range(start_size,len(Y))]

            # similarities = [(cosine_similarity(next_frame,new_label_features[j]),j) for j in range(start_size,len(new_label_features))]
            similarities = sorted(dist)
            # similarities= similarities[::-1]

            for s in similarities:
                if s[1] in frames_left:
                    next_frame = X[s[1]+1]

                    frames_left.remove(s[1])
                    idx_sequence[0].append(s[1])
                    break 

            sequence.append(next_frame)

        print idx_sequence

        for i in range(len(idx_sequence[0])-1):
            if(idx_sequence[0][i] + 1  == idx_sequence[0][i+1]):
                count+=1
        print count/(1.0*(len(idx_sequence[0])-1))
        
        if cost_val < 1e-4:
            break;

except KeyboardInterrupt:
    pass


# net = theanets.Regressor([len(X[0]),(100,"sigmoid"), len(X[0])])
# # train = np.array(X),np.array(Y,dtype = "int32")
# train = X[:size],Y[:size]
# print "training"
# net.train(train, algo='sgd', learning_rate=1e-4, momentum=0.9)

# # Show confusion matrices on the training/validation splits.
# print "predicting"
# pred = net.predict(X[size:])
# cv2.imshow("y",X[1].reshape(64,64).astype("uint8"))
# cv2.waitKey(1000)
# pred = probs(X_test)
# pred = np.array(pred,dtype = "int32")

size = 11



# for i in range(len(X_train)):
#     truth = Y_train[i]
#     pred = probs(np.array([X_train[i]]))
#     pred = np.array(255*pred,dtype = "uint8")

#     # cv2.imwrite("pred"+str(i)+".jpg",pred.reshape(64,64).astype("uint8"))
#     # cv2.imwrite("y"+str(i)+".jpg",truth.reshape(64,64).astype("uint8"))


#     similarities = [(cosine_similarity(pred,Y_train[j]),j) for j in range(len(Y_train))]
#     similarities = sorted(similarities)
#     similarities= similarities[::-1]


#     if similarities[0][1] != i:
#         if similarities[0][1] == i+1:
#             count+=1
#     else:
#         if similarities[1][1]== i+1:
#             count+=1
#     l = [s[1] for s in similarities]

#     print i , l



# for i in range(len(X_test)):
#     truth = Y_test[i]
#     pred = probs(np.array([X_test[i]]))
#     pred = np.array(255*pred,dtype = "uint8")

#     # cv2.imwrite("pred"+str(i)+".jpg",pred.reshape(64,64).astype("uint8"))
#     # cv2.imwrite("y"+str(i)+".jpg",truth.reshape(64,64).astype("uint8"))


#     similarities = [(cosine_similarity(pred,Y_test[j]),j+size) for j in range(len(Y_test))]
#     similarities = sorted(similarities)
#     similarities= similarities[::-1]


#     if similarities[0][1] != i+size:
#     	if similarities[0][1] == i+size+1:
#     		count+=1
#     else:
#     	if similarities[1][1]== i+size+1:
#     		count+=1
#     l = [s[1] for s in similarities[:3]]

#     print i+size , l

# print count/(20-1)

#
