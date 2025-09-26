## CS7180 - Advanced Perception
## DATE: 09/21/2025
## Gautham Ramkumar & Sai Vamsi Rithvik Allanka

import tensorflow as tf
from tensorflow import keras

def ssim_loss(y_true, y_pred):
    return 1.0 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))

## L1 Loss is provided by Keras directly as MeanAbsoluteError
l1_loss = keras.losses.MeanAbsoluteError()