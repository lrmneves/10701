import sys
import numpy as np

import caffe

import caffecnn

__author__ = "Zheng Chen (zhengc1@cs.cmu.edu)"
__version__ = "v1.0"

"""Extract features of moving car video using Caffe pre-trained CNN"""

if __name__ == "__main__":
    caffe_root = "/home/public/caffe"
    model_name = "bvlc_reference_caffenet"
    feature_size = 4096 # the size for caffenet (alexnet) is 4096
    data_path = "/home/public/10701/feature/car.npy"
    frame_ids = range(414)   # list of frames to extract feature

    # load model
    trainedcnn = caffecnn.TrainedCNN(caffe_root=caffe_root,
        model_name=model_name,
        cpu_mode=False,
        transformer=None,
        data_mean=None)

    # load data
    data = np.load(data_path)
    # (414, 720, 1280) -> (414, 720 * 1280)
    #data = data.reshape(data.shape[0], data.shape[1] * data.shape[2])
    # (414, 720, 1280) -> (414, 720, 1280, 3)
    data = data.reshape(data.shape + (1,)).repeat(3, 3)
    feature_set = np.zeros([len(frame_ids), feature_size])
    framecnt = 0
    for frameid in frame_ids:
        image = data[frameid]
        res = {}
        res["top_k"] = trainedcnn.predict_image(image)
        res["feature"] = trainedcnn.get_layerdata()[0].flatten()
        #res["label"] = trainedcnn.get_label(res["top_k"])
        feature_set[framecnt] = res["feature"]
        framecnt = framecnt + 1
    #import pdb; pdb.set_trace()
    if len(sys.argv) == 1:
        fname = "car_feature.npy"
    else:
        fname = sys.argv[1] + ".npy"
    with open(fname, "w") as outf:
        np.save(outf, feature_set)

