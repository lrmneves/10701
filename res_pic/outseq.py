import numpy as np
import cv2

__author__ = "Zheng Chen (zhengc1@cs.cmu.edu)"
__version__ = "v1.0"

"""Output image sequence of given predicted index sequence """

if __name__ == "__main__":
    # load data, (20, 10000, 64, 64) -> (10000, 20, 64, 64)
    data = np.load("/home/public/mnist_test_seq.npy").transpose(1, 0, 2, 3)

    # metric learning
    need_base_1_to_0 = True
    video_ids = [1011, 1001, 1009]
    frame_seqs = [
        range(1, 21), # id_1011: perfect
        [1,2,3,4,5,6,7,8,9,10,11,12,16,17,13,15,20,19,18,14],   # id_1001: not bad
        [1,20,11,10,9,12,13,17,19,2,4,6,7,14,15,8,5,3,18,16],   # id_1009: bad
    ]
    # simple NN
    need_base_1_to_0 = True
    video_ids = [0]
    frame_seqs = [
        [0,1,2,3,4,5,6,7,8,9,10,11,17,18,15,16,14,13,12,19]
    ]
    # googlenet
    need_base_1_to_0 = False
    video_ids = [0, 1, 2, 7]
    frame_seqs = [
        [0,1,2,14,11,12,13,10,15,19,6,16,18,5,8,3,4,9,17,7],    # id_0: bad
        [0,2,4,3,5,6,8,7,9,10,11,12,13,14,16,19,17,18,15,1],    # id_1: not bad
        [0,11,7,4,3,2,1,14,13,12,10,8,9,17,19,18,6,5,15,16],    # id_2: bad
        [0,1,3,4,5,6,7,8,9,10,11,13,14,15,19,17,16,18,12,2],    # id_7: best
    ]
    # lstm mnist run-1
    need_base_1_to_0 = False
    video_ids = [0]
    frame_seqs = [
        [0,1,2,3,4,5,6,7,8,9,10,16,17,15,14,18,11,13,12,19],
    ]
    # alexnet mnist
    need_base_1_to_0 = False
    video_ids = [2, 9, 12, 22, 35]
    frame_seqs = [
        [0,19,17,16,18,9,8,10,11,7,6,5,4,3,2,1,14,15,13,12],    # id_2: bad
        [0,1,6,3,12,14,13,11,10,9,7,8,5,4,2,19,15,17,18,16],    # id_9: bad
        [0,1,2,3,4,9,6,7,8,5,10,11,12,13,14,15,18,17,16,19],    # id_12: best
        [0,2,3,4,5,6,7,8,9,10,12,13,11,14,16,15,19,17,18,1],    # id_22: not bad
        [0,19,17,16,15,18,13,12,10,11,14,1,2,3,5,6,7,8,9,4],    # id_35: bad
    ]
    # alexnet cosine similarity car video
    # load data, (414, 720, 1280) -> (1, 414, 720, 1280)
    data = np.load("/home/public/10701/feature/car.npy")
    data = data.reshape((1,) + data.shape)
    need_base_1_to_0 = False
    video_ids = [0]
    frame_seqs = [
        [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19],
    ]
    #frame_seqs = [
    #    [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,26,25,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,47,48,49,46,27]
    #]
    # alexnet lstm car video
    need_base_1_to_0 = False
    video_ids = [0]
    frame_seqs = [
        [0,1,2,3,4,5,6,7,8,9,10,15,19,16,12,11,13,17,18,14],
    ]

    pic_height = data.shape[2]
    pic_width = data.shape[3]
    v_gap = 3
    v_gap_color = 255
    frame_size = 20
    #frame_size = 50

    video_cnt = 0
    for video_id in video_ids:
        correct_pic = np.zeros([pic_height, (pic_width + v_gap) * frame_size])
        frame_cnt = 0
        for cor_id in range(frame_size):
            # Append the correct pictures
            for lineno in range(pic_height):
                s = frame_cnt * (pic_width + v_gap)
                e = s + pic_width - 1
                correct_pic[lineno][s:e+1] = data[video_id][cor_id][lineno]
                correct_pic[lineno][e+1:e+6] = v_gap_color
            frame_cnt = frame_cnt + 1

        pred_pic = np.zeros([pic_height, (pic_width + v_gap) * frame_size])
        frame_cnt = 0
        for pred_id in frame_seqs[video_cnt]:
            if need_base_1_to_0:
                pred_id = pred_id - 1   # base 1 -> base 0
            # Append the predict pictures
            for lineno in range(pic_height):
                s = frame_cnt * (pic_width + v_gap)
                e = s + pic_width - 1
                pred_pic[lineno][s:e+1] = data[video_id][pred_id][lineno]
                pred_pic[lineno][e+1:e+6] = v_gap_color
            frame_cnt = frame_cnt + 1

        cv2.imwrite("pred_" + str(video_id) + ".jpg", pred_pic.astype("uint8"))
        cv2.imwrite("y_" + str(video_id) + ".jpg", correct_pic.astype("uint8"))
    
        video_cnt = video_cnt + 1

