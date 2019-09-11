import os
import pickle

import keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from sl_metric import evaluate_polygonal_results
from ssd_metric import fscore
from tbpp_model import TBPP512_dense
import tbpp_utils
from utils.bboxes import rbox3_to_polygon
from utils.model import load_weights, calc_memory_usage
from utils.vis import plot_box

from data_synthtext import GTUtility
with open('../gt_util_synthtext.pkl', 'rb') as f:
    gt_util = pickle.load(f)

gt_util_train, gt_util_val = gt_util.split(0.9)

# TextBoxes++ + DenseNet
model = TBPP512_dense(softmax=False)
weights_path = '/home/eugene/_MODELS/scene_text/201906190710_dsodtbpp512fl_synthtext/weights.022.h5'
confidence_threshold = 0.35
plot_name = 'dsodtbpp512fl_sythtext'

load_weights(model, weights_path)

prior_util = tbpp_utils.PriorUtil(model)

_, inputs, images, data = gt_util_val.sample_random_batch(1024)
inputs, images, data = inputs[:10], images[:10], data[:10]
preds = model.predict(inputs, batch_size=1, verbose=1)

for i in range(16):
    res = prior_util.decode(preds[i], confidence_threshold, fast_nms=False)
    bbox = res[:, 0:4]
    quad = res[:, 4:12]
    rbox = res[:, 12:17]
    # print(bbox)

    plt.figure(figsize=[8] * 2)
    plt.imshow(images[i])
    ax = plt.gca()
    for j in range(len(bbox)):
        # ax.add_patch(plt.Polygon(p, fill=False, edgecolor='r', linewidth=1))
        plot_box(bbox[j] * 512, box_format='xyxy', color='b')
        plot_box(np.reshape(quad[j], (-1, 2)) * 512, box_format='polygon', color='r')
        plot_box(rbox3_to_polygon(rbox[j]) * 512, box_format='polygon', color='g')
        plt.plot(rbox[j, [0, 2]] * 512, rbox[j, [1, 3]] * 512, 'oc', markersize=4)
    # prior_util.plot_gt()
    # prior_util.plot_results(res)
    plt.axis('off')
    plt.show()

steps = np.arange(0.05, 1, 0.05)

fmes_grid = np.zeros((len(steps)))

for i, t in enumerate(steps):
    results = [prior_util.decode(p, t) for p in preds]
    TP, FP, FN = evaluate_polygonal_results([g[:, 0:8] for g in data], [d[:, 4:12] for d in results])
    recall = TP / (TP + FN)
    precision = TP / (TP + FP)
    fmes = fscore(precision, recall)
    fmes_grid[i] = fmes
    print('threshold %.2f f-measure %.2f' % (t, fmes))

max_idx = np.argmax(fmes_grid)
print(steps[max_idx], fmes_grid[max_idx])
plt.figure(figsize=[12, 6])
plt.plot(steps, fmes_grid)
plt.plot(steps[max_idx], fmes_grid[max_idx], 'or')
plt.xticks(steps)
plt.grid()
plt.xlabel('threshold')
plt.ylabel('f-measure')
plt.show()

batch_size = 32

max_samples = gt_util_val.num_samples
max_samples = batch_size * 32

test_gt = []
test_results = []

for i in tqdm(range(int(np.ceil(max_samples / batch_size)))):
    inputs, data = gt_util_val.sample_batch(batch_size, i)
    preds = model.predict(inputs, batch_size, verbose=0)
    res = [prior_util.decode(p, confidence_threshold) for p in preds]
    test_gt.extend(data)
    test_results.extend(res)

TP, FP, FN = evaluate_polygonal_results([g[:, 0:8] for g in test_gt], [d[:, 4:12] for d in test_results])
recall = TP / (TP + FN)
precision = TP / (TP + FP)
fmes = fscore(precision, recall)

print('samples train     %i' % (gt_util_train.num_samples))
print('samples val       %i' % (gt_util_val.num_samples))

print('samples           %i' % (max_samples))
print('threshold         %0.3f' % (confidence_threshold))
print('precision         %0.3f' % (precision))
print('recall            %0.3f' % (recall))
print('f-measure         %0.3f' % (fmes))

trainable_count = int(np.sum([K.count_params(p) for p in set(model.trainable_weights)]))
non_trainable_count = int(np.sum([K.count_params(p) for p in set(model.non_trainable_weights)]))

print('trainable parameters     %10i' % (trainable_count))
print('non-trainable parameters %10i' % (non_trainable_count))
calc_memory_usage(model)
