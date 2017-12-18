import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import cv2

import matplotlib
matplotlib.use("Agg")
import matplotlib.backends.backend_agg as agg
import matplotlib.pylab as plt

from config import config


# Original By: Comma.ai
# ************************************************
# ***** get perspective transform for images *****
from skimage import transform as tf_

rsrc = \
    [[43.45456230828867, 118.00743250075844],
     [104.5055617352614, 69.46865203761757],
        [114.86050156739812, 60.83953551083698],
        [129.74572757609468, 50.48459567870026],
        [132.98164627363735, 46.38576532847949],
        [301.0336906326895, 98.16046448916306],
        [238.25686790036065, 62.56535881619311],
        [227.2547443287154, 56.30924933427718],
        [209.13359962247614, 46.817221154818526],
        [203.9561297064078, 43.5813024572758]]
rdst = \
    [[10.822125594094452, 1.42189132706374],
     [21.177065426231174, 1.5297552836484982],
        [25.275895776451954, 1.42189132706374],
        [36.062291434927694, 1.6376192402332563],
        [40.376849698318004, 1.42189132706374],
        [11.900765159942026, -2.1376192402332563],
        [22.25570499207874, -2.1376192402332563],
        [26.785991168638553, -2.029755283648498],
        [37.033067044190524, -2.029755283648498],
        [41.67121717733509, -2.029755283648498]]

tform3_img = tf_.ProjectiveTransform()
tform3_img.estimate(np.array(rdst), np.array(rsrc))


def perspective_tform(x, y):
    p1, p2 = tform3_img((x, y))[0]
    return p2, p1

# ***** functions to draw lines *****
# Hard coded adjustment of 150 for now, to center the steering visualization.


def draw_pt(img, x, y, color, sz=2):
    row, col = perspective_tform(x, y)
    if row >= 0 and row < img.shape[0] and\
       col >= 0 and col < img.shape[1]:
        img[(int(row) - sz):(int(row) + sz), (int(col + 150) - sz):(int(col + 150) + sz)] = color


def draw_path(img, path_x, path_y, color):
    for x, y in zip(path_x, path_y):
        draw_pt(img, x, y, color)

# ***** functions to draw predicted path *****


def calc_curvature(v_ego, angle_steers, angle_offset=0):
    deg_to_rad = np.pi / 180.
    slip_fator = 0.0014  # slip factor obtained from real data
    steer_ratio = 15.3  # from http://www.edmunds.com/acura/ilx/2016/road-test-specs/
    wheel_base = 2.67   # from http://www.edmunds.com/acura/ilx/2016/sedan/features-specs/

    angle_steers_rad = (angle_steers - angle_offset)  # * deg_to_rad
    curvature = angle_steers_rad / (steer_ratio * wheel_base * (1. + slip_fator * v_ego**2))
    return curvature


def calc_lookahead_offset(v_ego, angle_steers, d_lookahead, angle_offset=0):
    #*** this function returns the lateral offset given the steering angle, speed and the lookahead distance
    curvature = calc_curvature(v_ego, angle_steers, angle_offset)

    # clip is to avoid arcsin NaNs due to too sharp turns
    y_actual = d_lookahead * np.tan(np.arcsin(np.clip(d_lookahead * curvature, -0.999, 0.999)) / 2.)
    return y_actual, curvature


def draw_path_on(img, speed_ms, angle_steers, color=(0, 0, 255)):
    path_x = np.arange(0., 100.1, 0.20)
    path_y, _ = calc_lookahead_offset(speed_ms, angle_steers, path_x)
    draw_path(img, path_x, path_y, color)

# ************************************************


def test_fn(data, data_dir):
    def _convert_image(data_set):
        path = data_dir + data_set['image']
        data_set['image'] = tf.to_float(tf.image.decode_image(tf.read_file(path)))
        return data_set

    dataset = tf.data.Dataset.from_tensor_slices(data)
    dataset = dataset.map(_convert_image)
    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()
    return next_element

data_dir = config['bag4']
file_name = r'predictions.pickle'

with open(file_name, 'rb') as f:
    data = pickle.load(f)

data_list = ['image', 'steer_pred', 'steer_label']

reformat = lambda x: np.concatenate(np.array(data[x]).reshape(7,))

for key in data_list:
    data[key] = reformat(key)

red = (255, 0, 0)
blue = (0, 0, 255)
speed_ms = 5  # log['speed'][i]


# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

with tf.Session() as sess:
    test_next = test_fn(data, data_dir)
    residuals_squared = []
    while True:
        try:
            data = sess.run(test_next)
            actual_steers = data['steer_label'][0]
            predicted_steers = data['steer_pred'][0]
            img = data['image']

            residuals_squared.append(actual_steers - predicted_steers)
            print(actual_steers, predicted_steers)

            draw_path_on(img, speed_ms, actual_steers / 2.0)
            draw_path_on(img, speed_ms, predicted_steers / 2.0, (255, 0, 0))
            out.write(np.uint8(img))

        except tf.errors.OutOfRangeError:
            out.release()
            break

    print("RMSE: {}".format(np.sqrt(np.mean(residuals_squared))))
