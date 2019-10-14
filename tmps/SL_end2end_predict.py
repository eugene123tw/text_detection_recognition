import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import cv2
import pickle
import tensorflow as tf

from sl_model import DSODSL512
from sl_utils import PriorUtil
from ssd_data import InputGenerator
from ssd_data import preprocess
from sl_utils import rbox_to_polygon
from utils.vis import plot_box

from utils.model import load_weights

weights_path = '/home/eugene/_MODELS/scene_text/seglink_ssd512_synthtext/weights.012.h5'
segment_threshold = 0.55
link_threshold = 0.45
plot_name = 'dsodsl512_crnn_sythtext'

sl_graph = tf.Graph()
with sl_graph.as_default():
    sl_session = tf.Session()
    with sl_session.as_default():
        model = DSODSL512()
        prior_util = PriorUtil(model)
        load_weights(model, weights_path)

image_size = model.image_size

inputs = []
images = []
images_orig = []
data = []

for img_path in glob.glob('/home/eugene/_DATASETS/porsche_fonts3/_test_images/*.jpg'):
    img = cv2.imread(img_path)
    images_orig.append(np.copy(img))
    inputs.append(preprocess(img, image_size))
    h, w = image_size
    img = cv2.resize(img, (w, h), cv2.INTER_LINEAR).astype('float32')  # should we do resizing
    img = img[:, :, (2, 1, 0)]  # BGR to RGB
    img /= 255
    images.append(img)

inputs = np.asarray(inputs)


for i in range(len(inputs)):
    plt.imshow(images[i])
    with sl_graph.as_default():
        with sl_session.as_default():
            preds = model.predict(inputs[i:i+1], batch_size=1, verbose=0)
            res = prior_util.decode(preds[0], segment_threshold, link_threshold)
            rboxes = res[:, :5]

            bh = rboxes[:, 3]
            rboxes[:, 2] += bh * 0.1
            rboxes[:, 3] += bh * 0.2

            boxes = np.asarray([rbox_to_polygon(r) for r in rboxes])
            boxes = np.flip(boxes, axis=1)  # TODO: fix order of points, why?
            boxes = np.reshape(boxes, (-1, 8))

            boxes_mask = np.array([not (np.any(b < 0 - 10) or np.any(b > 512 + 10)) for b in boxes])  # box inside image
            # boxes_mask = np.logical_and(boxes_mask, [b[2] > 0.8*b[3] for b in rboxes]) # width > height, in square world

            boxes = boxes[boxes_mask]
            rboxes = rboxes[boxes_mask]
            if len(boxes) == 0:
                boxes = np.empty((0, 8))

            # plot boxes
            for box in boxes:
                c = 'rgby'
                for i in range(4):
                    x, y = box[i * 2:i * 2 + 2]
                    plt.plot(x, y, c[i], marker='o', markersize=4)
                plot_box(box, 'polygon')

            plt.show()