import numpy as np, re, theanets
import scipy.io
import cv2

dataset = np.load("data/mnist_test_seq.npy")

# scipy.io.savemat("data.mat",mdict={'dataset': dataset})
first_clip = []
second_clip = []

for frame in dataset:
	# if(len(first_clip) < len(dataset)/2):
	first_clip.append(frame[0].flatten())
	# else:
	# 	second_clip.append(frame[0].flatten())

x = np.array(first_clip).astype('f')
y = []
BATCH_SIZE = len(dataset)/2
STEPS = 2
IN_SIZE = first_clip[0]
OUT_SIZE = len(dataset)

for i in range(len(first_clip)):
	y.append(i+1%BATCH_SIZE)




print x.shape[1]

# net = theanets.recurrent.Classifier([len(first_clip[0]),len(first_clip[0])])


y = np.array(y).astype("int32")


# for train, valid in net.itertrain([x,y], optimize='adadelta', patience=100000, batch_size=BATCH_SIZE):
#     print('training loss:', train['loss'])

#     print('training acc :', train['acc'])
# # # # create a model to train: input -> gru -> relu -> softmax.
# #, (1000, 'relu')


train = x,np.array(y,dtype = "int32")
net.train(train, algo='sgd', learning_rate=1e-4, momentum=0.9)

# #
# # # # train the model iteratively; draw a sample after every epoch.
# # #	seed = txt.encode(txt.text[300017:300050])
# # for tm, _ in net.itertrain(txt.classifier_batches(100, 32), momentum=0.9):
# #    		net.predict_sequence(first_clip, 40))
