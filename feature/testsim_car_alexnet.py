import numpy as np
import scipy.spatial.distance as sdist

"""A test on car video using caffenet feature and cosine similarity """

if __name__ == "__main__":
    # configuration
    feature_path = "/home/public/10701/feature/car_feature_414.npy"
    feature_size = 4096
    train_frame_limit = 50
    
    # load feature, (414, 4096)
    feature = np.load(feature_path)
    train_frame_size = min(train_frame_limit, feature.shape[0])
    result_set = np.zeros([train_frame_size])
    # calculate the distances
    nearids = sdist.squareform(sdist.pdist(
        feature[0:train_frame_size], 'cosine')).argsort()
    result_set[0] = 0  # Always set the first one as known
    visit_set = set([0])
    for frameid in range(1, train_frame_size):
        lastid = result_set[frameid - 1]
        for nearid in nearids[lastid]:
            if nearid not in visit_set:
                result_set[frameid] = nearid
                visit_set.add(nearid)
                break
    print result_set

