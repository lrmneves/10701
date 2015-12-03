import numpy as np, re, theanets
from sklearn.metrics import confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
import cv2

# dataset = np.load("new_data.npy")
# new_data = []
# for frame in dataset:
# 	new_frames = []
# 	for im in range(len(frame)):
# 		new_frames.append(frame[im].flatten())
# 	new_data.append(new_frames)


# X = [f[0] for f in dataset]
# Y = X[1:]
# X = X[:-1]
# np.save("X.npy",np.array(X))
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

size = 10

X,Y = np.load("X.npy"),np.load("Y.npy")


net = theanets.Regressor([len(X[0]),(100,"relu"), len(X[0])])
# train = np.array(X),np.array(Y,dtype = "int32")
train = X[:size],Y[:size]
print "training"
net.train(train, algo='sgd', learning_rate=1e-3, momentum=0.9)

# Show confusion matrices on the training/validation splits.
print "predicting"
pred = net.predict(X[size:])
# cv2.imshow("y",X[1].reshape(64,64).astype("uint8"))
# cv2.waitKey(1000)


count = 0.0



seen = set()

sequence = []
for i in range(len(pred-1)):

	cv2.imwrite("pred"+str(i)+".jpg",pred[i].reshape(64,64).astype("uint8"))
	cv2.imwrite("y"+str(i)+".jpg",X[i+10].reshape(64,64).astype("uint8"))

	similarities = [(cosine_similarity(pred[i],X[j]),j) for j in range(10,len(X)) if not j in seen]
	similarities = sorted(similarities)
	similarities= similarities[::-1]


	if similarities[0][1] != i+size:
		if similarities[0][1] == i+size+1:
			count+=1
	else:
		if similarities[1][1]== i+size+1:
			count+=1
	l = [s[1] for s in similarities]
	seen.add(l[0])

	sequence.append(l[0])
	print i+size , l

print count/(20-size-1)

print sequence
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