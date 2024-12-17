import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import MaxPool2D, AvgPool2D


def pool(input, pool_type, data_format='channels_last'):
    """Apply pooling operation (via Tensorflow) to input Numpy array x.
    x should be 4-dimensional: N x W x H x C ('channels_last') or N x C x W x H ('channels_first')
    Pooling is applied on W and H dimensions.

    """
    x = np.copy(input)
    if len(x.shape) == 2:
        x = np.expand_dims(x, (0, -1))
    elif len(x.shape) == 3:
        x = np.expand_dims(x, 0)
        
    pool_op = {
        'no_pooling': lambda x: tf.convert_to_tensor(x),
        'max_4': MaxPool2D(pool_size=(4, 4), strides=(2, 2), data_format=data_format),
        'max_16': MaxPool2D(pool_size=(16, 16), strides=(4, 4), data_format=data_format),
        'avg_4': AvgPool2D(pool_size=(4, 4), strides=(2, 2), data_format=data_format),
        'avg_16': AvgPool2D(pool_size=(16, 16), strides=(4, 4), data_format=data_format),
    }[pool_type]

    return pool_op(x.astype("float32")).numpy()
