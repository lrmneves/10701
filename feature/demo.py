import os
import matplotlib.pyplot as plt

import caffe
from caffe.io import load_image

import caffecnn

__author__ = "Zheng Chen (zhengc1@cs.cmu.edu)"
__version__ = "v1.0"

"""A demo of the cafecnn.TrainedCNN"""

def classify_image(trainedcnn, image_path, show_image=False):
    """Use the trained CNN to calssify an image

    :returns: A dict, "top_k" are the ids of top n result,
              "label" are labels of the top n result,
              "feature" are the input of the classifier in CNN
    """
    res = {}
    res["top_k"] = trainedcnn.predict_image(load_image(image_path))
    res["feature"] = trainedcnn.get_layerdata()[0].flatten()
    res["label"] = trainedcnn.get_label(res["top_k"])
    if show_image:
        # show the images
        plt.imshow(trainedcnn.get_data(0))
        plt.show()

    return res
    
if __name__ == "__main__":
    # configuration
    caffe_root = "/home/public/caffe/"
    model_name = "bvlc_googlenet"
    show_image = False
    
    # set the plot parameters
    if show_image:
        plt.rcParams["figure.figsize"] = (10, 10)
        plt.rcParams["image.interpolation"] = "nearest"
        plt.rcParams["image.cmap"] = "gray"
    
    # load model
    trainedcnn = caffecnn.TrainedCNN(caffe_root=caffe_root,
        model_name=model_name,
        cpu_mode=False,
        transformer=None,
        data_mean=None)
    
    # load data
    fb_img_path = os.path.join(caffe_root, "examples/images/fish-bike.jpg")
    cat_img_path = os.path.join(caffe_root, "examples/images/cat.jpg")
    cat_gray_img_path = os.path.join(caffe_root, "examples/images/cat_gray.jpg")
    
    # classify images
    res = classify_image(trainedcnn, fb_img_path, show_image=show_image)
    res = classify_image(trainedcnn, cat_img_path, show_image=show_image)
    res = classify_image(trainedcnn, cat_gray_img_path, show_image=show_image)
