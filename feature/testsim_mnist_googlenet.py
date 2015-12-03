import numpy as np
import scipy.spatial.distance as sdist

"""A test on MNIST using caffe googlenet feature and cosine similarity """

if __name__ == "__main__":
    # configuration
    feature_path = "/home/public/10701/feature/mnist_feature_100_20.npy"
    feature_size = 1024
    train_video_limit = 10
    train_frame_limit = 20
    
    # load feature, (100, 20, 1024)
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

