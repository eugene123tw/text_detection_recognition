import numpy as np
import matplotlib.pyplot as plt
import keras
import time
import os
import pickle

from tbpp_model import TBPP512, TBPP512_dense
from tbpp_utils import PriorUtil
import ssd_data
from tbpp_training import TBPPFocalLoss
from utils.model import load_weights
from utils.training import Logger

from data_synthtext import GTUtility
with open('../gt_exp.pkl', 'rb') as f:
# with open('gt_util_synthtext_seglink.pkl', 'rb') as f:
    gt_util = pickle.load(f)

gt_util_train, gt_util_val = gt_util.split(0.9)

# TextBoxes++ + DenseNet
model = TBPP512_dense(softmax=False)
# weights_path = '/home/eugene/_MODELS/scene_text/201906190710_dsodtbpp512fl_synthtext/weights.022.h5'
weights_path = None
freeze = []
batch_size = 6
experiment = 'dsodtbpp512fl_synthtext'

prior_util = PriorUtil(model)

if weights_path is not None:
    load_weights(model, weights_path)


epochs = 100
initial_epoch = 0

gen_train = ssd_data.InputGenerator(gt_util_train, prior_util, batch_size, model.image_size)
gen_val = ssd_data.InputGenerator(gt_util_val, prior_util, batch_size, model.image_size)

for layer in model.layers:
    layer.trainable = not layer.name in freeze

checkdir = './checkpoints/' + time.strftime('%Y%m%d%H%M') + '_' + experiment
if not os.path.exists(checkdir):
    os.makedirs(checkdir)

# with open(checkdir+'/source.py','wb') as f:
#     source = ''.join(['# In[%i]\n%s\n\n' % (i, In[i]) for i in range(len(In))])
#     f.write(source.encode())

#optim = keras.optimizers.SGD(lr=1e-3, momentum=0.9, decay=0, nesterov=True)
optim = keras.optimizers.Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=0.001, decay=0.0)

# weight decay
regularizer = keras.regularizers.l2(5e-4) # None if disabled
#regularizer = None
for l in model.layers:
    if l.__class__.__name__.startswith('Conv'):
        l.kernel_regularizer = regularizer

loss = TBPPFocalLoss(lambda_conf=10000.0, lambda_offsets=1.0)

model.compile(optimizer=optim, loss=loss.compute, metrics=loss.metrics)

print(checkdir.split('/')[-1])

history = model.fit_generator(
        gen_train.generate(debug=True),
        steps_per_epoch=int(gen_train.num_batches/4),
        epochs=epochs,
        verbose=1,
        callbacks=[
            keras.callbacks.ModelCheckpoint(checkdir+'/weights.{epoch:03d}.h5', verbose=1, save_weights_only=True),
            Logger(checkdir),
            #LearningRateDecay()
        ],
        validation_data=gen_val.generate(debug=True),
        validation_steps=gen_val.num_batches,
        class_weight=None,
        max_queue_size=1,
        workers=1,
        #use_multiprocessing=False,
        initial_epoch=initial_epoch,
        #pickle_safe=False, # will use threading instead of multiprocessing, which is lighter on memory use but slower
        )