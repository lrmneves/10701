import numpy as np
import scipy.spatial.distance as sdist

__author__ = "Zheng Chen (zhengc1@cs.cmu.edu)"
__version__ = "v1.0"

"""A test on MNIST using caffenet feature and cosine similarity """

if __name__ == "__main__":
    # configuration
    feature_path = "/home/public/10701/feature/mnist_feature_50_20_alex.npy"
    feature_size = 4096
    train_video_limit = 10
    train_frame_limit = 20
    
    # load feature, (50, 20, 4096)
    feature = np.load(feature_path)
    train_video_size = min(train_video_limit, feature.shape[0])
    train_frame_size = min(train_frame_limit, feature.shape[1])
    result_set = np.zeros([train_video_size, train_frame_size])
    for videoid in range(train_video_size):
        # calculate the distances
        nearids = sdist.squareform(sdist.pdist(
            feature[videoid], 'cosine')).argsort()
        result_set[videoid][0] = 0  # Always set the first one as known
        visit_set = set([0])
        for frameid in range(1, train_frame_size):
            lastid = result_set[videoid][frameid - 1]
            for nearid in nearids[lastid]:
                if nearid not in visit_set:
                    result_set[videoid][frameid] = nearid
                    visit_set.add(nearid)
                    break
        print result_set[videoid]

