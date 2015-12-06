import os
import time
import numpy as np

import caffe

__author__ = "Zheng Chen (zhengc1@cs.cmu.edu)"
__version__ = "v1.0"

class TrainedCNN(object):
    """Caffe pre-trained CNN wrapper
    """
    def __init__(self, caffe_root, model_name="bvlc_googlenet",
        cpu_mode=True, transformer=None, data_mean=None):
        """Load a pre-trained CNN and setup the pre-processing transformer

        :param caffe_root: The root path of Caffe
        :param model_name: The name of pre-trained model in Caffe Model Zoo
        :param cpu_mode: Set CPU/GPU mode
        :param transformer: Pprocessing transf, set None to use default
        :param data_mean: Mean of data the model trained on, default ilsvrc12
        :returns: The pre-trained CNN wrapper
        :raises IOError
        """

        if not os.path.isdir(caffe_root):
            raise IOError("caffe root %s is not a directory" % caffe_root)
        self.caffe_root = caffe_root

        # set model path
        model_path = os.path.join(caffe_root, "models", model_name,
             model_name + ".caffemodel")
        model_deploy_path = os.path.join(caffe_root, "models", model_name,
            "deploy.prototxt")

        # set the CPU/GPU mode
        if cpu_mode:
            caffe.set_mode_cpu()
        else:
            caffe.set_mode_gpu()

        # load the model
        if not os.path.isfile(model_path) \
            or not os.path.isfile(model_deploy_path):
            raise IOError("Cannot load model from %s" % model_path)

        self.net = caffe.Net(model_deploy_path, model_path, caffe.TEST)

        # data transformer for pre-processing
        if transformer is None:
            transformer = caffe.io.Transformer({
                "data": self.net.blobs["data"].data.shape
            })
            transformer.set_transpose("data", (2, 0, 1))
            # the trained model has channels in BGR order instead of RGB
            transformer.set_channel_swap("data", (2, 1, 0))
            # the trained model operates on images in [0,255] instead of [0,1]
            transformer.set_raw_scale("data", 255)
            if data_mean is None:
                mean_path = os.path.join(caffe_root, "python/caffe/imagenet/" \
                    "ilsvrc_2012_mean.npy")
                if not os.path.isfile(mean_path):
                    raise IOError("Cannot find %s" % mean_path)
                data_mean = np.load(mean_path).mean(1).mean(1)
            # should be the mean of data the model is trained on
            transformer.set_mean("data", data_mean)
        self.transf = transformer

        # optional: load labels
        self.load_labels()

    def predict_image(self, image, k=5):
        """Use the pre-trained CNN to classify an image, return top k result

        :param image: The image to be classified
        :param k: Return the top k result
        :returns: The index of the top k result (on ilsvrc12 data-set)
        """
        self.net.blobs["data"].data[...] = self.transf.preprocess("data", image)
        start = time.time()
        out = self.net.forward()
        print "Image classified in %.3f seconds" % (time.time() - start)

        top_k = self.net.blobs["prob"].data[0].flatten().argsort()[-1:-(k+1):-1]
        return top_k

    def get_layerdata(self, layer_id=-3):
        """Return the output of a layer

        :param layer_id: The id of layer, usually the last layer is probability
                         and the second last is loss/classifier, thus if you
                         want the input of the classifier it should be -3
        """
        layer_name = self.net.blobs.keys()[layer_id]
        return self.net.blobs[layer_name].data

    def get_label(self, label_ids):
        """Return the labels on ilsvrc12 data-set of the given label ids

        :param label_ids: A list of label ids
        :returns: A list of labels
        """
        if self.labels is None:
            print "No labels found! Use load_labels() to load the labels first."
            return []
        return self.labels[label_ids]

    def load_labels(self, label_path=None):
        """Load labels of the data-set

        :param label_path: The path of the label file, default ilsvrc12
        """
        if label_path is None:
            label_path = os.path.join(self.caffe_root, "data/" \
                "ilsvrc12/synset_words.txt")
        try:
            self.labels = np.loadtxt(label_path, str, delimiter = "\t")
        except:
            self.labels = None
            print "%s not found, labels are not loaded." % label_path
        
    def get_data(self, n):
        """Return the n'th data (image) in the model"""
        return self.transf.deprocess("data", self.net.blobs["data"].data[n])
