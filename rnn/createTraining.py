import numpy as np
from random import randint
path = "/home/public/10701/feature/car_feature_414.npy"
#path = "/home/public/10701/feature/mnist_feature_50_20_alex.npy"
X_full = np.load(path)
#X_t = X_full[0]
X_t = X_full[20:40]


MIN_LENGTH = 5


TRAIN_SIZE = 1000

X = []
Y = []



for i in range(TRAIN_SIZE):
	start = randint(0,len(X_t)-MIN_LENGTH - 2)
	end = randint(start + MIN_LENGTH,len(X_t)-2)
	pos_or_neg = randint(0, 1)
	if pos_or_neg == 0: #means its negative
		prediction =randint(0,len(X_t)-1)
		while prediction == end + 1:
			prediction =randint(0,len(X_t)-1)
		current = np.vstack([X_t[start:end],X_t[prediction]])
		X.append(current)
		Y.append(0)
	else:
		current = X_t[start:end+1]
		X.append(current)
		Y.append(1)

MAX_LENGTH = 20
#Padding the sequence
for i in range(len(X)):
	z = np.zeros((MAX_LENGTH-len(X[i]),len(X[i][0])))
	X[i] = np.vstack([X[i],z])



print np.mean(Y)

np.save("X_car_class.npy",np.array(X))
np.save("Y_car_class.npy",np.array(Y))
