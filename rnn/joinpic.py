import numpy as np
import cv2

__author__ = "Zheng Chen (zhengc1@cs.cmu.edu)"
__version__ = "v1.0"

"""Output image sequence of given predicted pictures """

if __name__ == "__main__":
    # load data, (20, 10000, 64, 64) -> (10000, 20, 64, 64)
    data = np.load("/home/public/mnist_test_seq.npy").transpose(1, 0, 2, 3)
    pic_height = 64
    pic_width = 64
    v_gap = 3
    v_gap_color = 255
    frame_size = 20
    need_base_1_to_0 = True
    video_ids = [0]

    # load predict pictures
    pic_num = 10
    pic_name_pattern = "pred%d.jpg"
    imgs = np.zeros([pic_num, pic_height, pic_width])
    for i in range(pic_num):
        imgs[i] = cv2.imread(pic_name_pattern % i, cv2.CV_LOAD_IMAGE_GRAYSCALE)

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
        for pred_id in range(frame_size - pic_num):
            if need_base_1_to_0:
                pred_id = pred_id - 1   # base 1 -> base 0
            # Append the training pictures
            for lineno in range(pic_height):
                s = frame_cnt * (pic_width + v_gap)
                e = s + pic_width - 1
                pred_pic[lineno][s:e+1] = data[video_id][pred_id][lineno]
                pred_pic[lineno][e+1:e+6] = v_gap_color
            frame_cnt = frame_cnt + 1
        for pred_id in range(pic_num):
            # Append the predict pictures
            for lineno in range(pic_height):
                s = frame_cnt * (pic_width + v_gap)
                e = s + pic_width - 1
                pred_pic[lineno][s:e+1] = imgs[pred_id][lineno]
                pred_pic[lineno][e+1:e+6] = v_gap_color
            frame_cnt = frame_cnt + 1

        cv2.imwrite("pred_" + str(video_id) + ".jpg", pred_pic.astype("uint8"))
        cv2.imwrite("y_" + str(video_id) + ".jpg", correct_pic.astype("uint8"))
    
        video_cnt = video_cnt + 1

