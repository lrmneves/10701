import numpy as np, re, theanets
from sklearn.metrics import confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
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

X,Y = np.load("X.npy"),np.load("Y.npy")

# Min/max sequence length
MIN_LENGTH = 10
MAX_LENGTH = 1
# Number of units in the hidden (recurrent) layer
N_HIDDEN = 10
# Number of training sequences in each batch
N_BATCH = 10
# Optimization learning rate

LEARNING_RATE = 0.01
# All gradients above this will be clipped
GRAD_CLIP = 1000
# How often should we check the output?
EPOCH_SIZE = 10
# Number of epochs to train the net
NUM_EPOCHS = 1000

NUM_FEATURES = len(X[0])
print NUM_FEATURES
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

	for l in range(MAX_LENGTH):
	
		X_train[b].append(X[b*MAX_LENGTH + l])
		Y_train.append(Y[b*MAX_LENGTH + l])
		try:
			X_test[b].append(X[b*MAX_LENGTH + l+9])
			Y_test.append(Y[b*MAX_LENGTH + l+9])
		except Exception:
			continue
# X_test = np.array([X[11:]])
# Y_test = np.array([Y[11:]])
mask_train = np.zeros((N_BATCH, MAX_LENGTH))
mask_test = np.zeros((N_BATCH, MAX_LENGTH))
for b in range(N_BATCH):
	mask_train[b,:len(X_train[b])] = 1
	mask_test[b,:len(X_test[b])] = 1

X_train = np.array(X_train)
Y_train = np.array(Y_train)

X_test= np.array(X_test)
Y_test = np.array(Y_test)
print len(Y_train)
print len(Y_train[0])
# print len(Y_train[0][0])

# First, we build the network, starting with an input layer
# Recurrent layers expect input of shape
# (batch size, max sequence length, number of features)
l_in = lasagne.layers.InputLayer(shape=(N_BATCH, MAX_LENGTH, NUM_FEATURES))
# The network also needs a way to provide a mask for each sequence.  We'll
# use a separate input layer for that.  Since the mask only determines
# which indices are part of the sequence for each batch entry, they are
# supplied as matrices of dimensionality (N_BATCH, MAX_LENGTH)
l_mask = lasagne.layers.InputLayer(shape=(N_BATCH, MAX_LENGTH))
# We're using a bidirectional network, which means we will combine two
# RecurrentLayers, one with the backwards=True keyword argument.
# Setting a value for grad_clipping will clip the gradients in the layer
# Setting only_return_final=True makes the layers only return their output
# for the final time step, which is all we need for this task
l_forward = lasagne.layers.RecurrentLayer(
    l_in, N_HIDDEN, mask_input=l_mask, grad_clipping=GRAD_CLIP,
    W_in_to_hid=lasagne.init.HeUniform(),
    W_hid_to_hid=lasagne.init.HeUniform(),
    nonlinearity=lasagne.nonlinearities.sigmoid, only_return_final=False)

l_backward = lasagne.layers.RecurrentLayer(
    l_in, N_HIDDEN, mask_input=l_mask, grad_clipping=GRAD_CLIP,
    W_in_to_hid=lasagne.init.HeUniform(),
    W_hid_to_hid=lasagne.init.HeUniform(),
    nonlinearity=lasagne.nonlinearities.tanh,
    only_return_final=False, backwards=True)

# Now, we'll concatenate the outputs to combine them.
l_concat = lasagne.layers.ConcatLayer([l_forward, l_backward])
# Our output layer is a simple dense connection, with 1 output unit
l_out = lasagne.layers.DenseLayer(
    l_concat, num_units=MAX_LENGTH*NUM_FEATURES, nonlinearity=lasagne.nonlinearities.tanh)

target_values = T.matrix('target_output')

# lasagne.layers.get_output produces a variable for the output of the net

network_output = lasagne.layers.get_output(l_out)
predictions = network_output
# cost = lasagne.objectives.squared_error(predictions, target_values).mean()
cost = T.mean(1 - T.batched_dot(predictions,target_values)/(tnorm(predictions)*tnorm(target_values)))
# print cost
# Retrieve all parameters from the network
all_params = lasagne.layers.get_all_params(l_out)
# Compute SGD updates for training
# print("Computing updates ...")
updates = lasagne.updates.adagrad(cost, all_params, LEARNING_RATE)
# Theano functions for training and computing cost
# print("Compiling functions ...")
train = theano.function([l_in.input_var, target_values, l_mask.input_var],
                            cost,
                            updates=updates)

get_pred = theano.function([l_in.input_var,l_mask.input_var],lasagne.layers.get_output(l_out))

compute_cost = theano.function(
    [l_in.input_var, target_values, l_mask.input_var], cost)

 # print("Training ...")
try:
    for epoch in range(NUM_EPOCHS):
        for _ in range(EPOCH_SIZE):
            
            train(X_train, Y_train, mask_train)
        cost_val = compute_cost(X_train, Y_train, mask_train)
        pred = get_pred(X_train,mask_train)
        print np.amax(pred)
        print("Epoch {} validation cost = {}".format(epoch, cost_val))
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
# # cv2.imshow("y",X[1].reshape(64,64).astype("uint8"))
# # cv2.waitKey(1000)


# count = 0.0

# for i in range(len(pred-1)):

# 	cv2.imwrite("pred"+str(i)+".jpg",pred[i].reshape(64,64).astype("uint8"))
# 	cv2.imwrite("y"+str(i)+".jpg",X[i+10].reshape(64,64).astype("uint8"))

# 	similarities = [(cosine_similarity(pred[i],X[j]),j) for j in range(10,len(X))]
# 	similarities = sorted(similarities)
# 	similarities= similarities[::-1]


# 	if similarities[0][1] != i+size:
# 		if similarities[0][1] == i+size+1:
# 			count+=1
# 	else:
# 		if similarities[1][1]== i+size+1:
# 			count+=1
# 	l = [s[1] for s in similarities[:3]]

# 	print i+size , l

# print count/(20-size-1)

# chars = re.sub(r'\s+', ' ', open('corpus.txt').read().lower())
# txt = theanets.recurrent.Text(chars, min_count=10)
# A = 1 + len(txt.alpha)  # of letter classes

# # # create a model to train: input -> gru -> relu -> softmax.
# net = theanets.recurrent.Classifier([A  A])

# # train the model iteratively; draw a sample after every epoch.
# seed = txt.encode(txt.text[300017:300050])
# print txt.classifier_batches(100, 32)
# for tm, _ in net.itertrain(txt.classifier_batches(100, 32), momentum=0.9):
#     print('{}|{} ({:.1f}%)'.format(
#         txt.decode(seed),
#         txt.decode(net.predict_sequence(seed, 40)),
#         100 * tm['acc']))