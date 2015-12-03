import sys
import numpy as np

import caffe

import caffecnn

__author__ = "Zheng Chen (zhengc1@cs.cmu.edu)"
__version__ = "v1.0"

"""Extract features of moving MNIST videos using Caffe pre-trained CNN"""

if __name__ == "__main__":
    caffe_root = "/home/public/caffe"
    model_name = "bvlc_googlenet"
    feature_size = 1024 # the size for googlenet is 1024
    data_path = "/home/public/mnist_test_seq.npy"
    video_ids = range(100)   # list of videoes to extract feature
    frame_ids = range(20)   # list of frames to extract feature

    # load model
    trainedcnn = caffecnn.TrainedCNN(caffe_root=caffe_root,
        model_name=model_name,
        cpu_mode=False,
        transformer=None,
        data_mean=None)

    # load data, (20, 10000, 64, 64) -> (10000, 20, 64, 64)
    data = np.load(data_path).transpose(1, 0, 2, 3)
    feature_set = np.zeros([len(video_ids), len(frame_ids), feature_size])
    videocnt = 0
    for videoid in video_ids:
        framecnt = 0
        for frameid in frame_ids:
            # should not use constants
            image = data[videoid][frameid].reshape(64, 64, 1).repeat(3, 2)
            # TODO: fill blank to 224 x 224 ?
            res = {}
            res["top_k"] = trainedcnn.predict_image(image)
            res["feature"] = trainedcnn.get_layerdata()[0].flatten()
            #res["label"] = trainedcnn.get_label(res["top_k"])
            feature_set[videocnt][framecnt] = res["feature"]
            framecnt = framecnt + 1
        videocnt = videocnt + 1
    #import pdb; pdb.set_trace()
    if len(sys.argv) == 1:
        fname = "mnist_feature.npy"
    else:
        fname = sys.argv[1] + ".npy"
    with open(fname, "w") as outf:
        np.save(outf, feature_set)

