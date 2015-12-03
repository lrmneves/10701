import numpy as np
from random import randint
X_full = np.load("/Users/lrmneves/workspace/Fall 2015/MachineLearning/finalProject/mnist_feature_100_20.npy")


X_t = X_full[0]


MIN_LENGTH = 2


TRAIN_SIZE = 20

X = []
Y = []



for i in range(TRAIN_SIZE):
	start = randint(0,len(X_t)-MIN_LENGTH - 2)
	end = randint(start + MIN_LENGTH,len(X_t)-2)
	pos_or_neg = randint(0, 1)
	if pos_or_neg == 0: #means its negative
		prediction =randint(0,len(X_t))
		while prediction != end + 1:
			prediction =randint(0,len(X_t))
		current = np.vstack([X_t[start:end],X_t[prediction]])
		X.append(current)
		Y.append(0)
	else:
		current = X_t[start:end+1]
		X.append(current)
		Y.append(1)

MAX_LENGTH = -1 
for i in range(len(X)):
    if MAX_LENGTH < len(X[i]):
        MAX_LENGTH = len(X[i])
#Padding the sequence
for i in range(len(X)):
	z = np.zeros((MAX_LENGTH-len(X[i]),len(X[i][0])))
	X[i] = np.vstack([X[i],z])



print np.mean(Y)

np.save("X_class.npy",np.array(X))
np.save("Y_class.npy",np.array(Y))