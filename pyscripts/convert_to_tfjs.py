import cv2
import numpy as np
import matplotlib.pyplot as plt
from local_utils import detect_lp
from os.path import splitext,basename
from keras.models import model_from_json,load_model
import glob
import time

import tensorflowjs as tfjs
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# import tensorflow as tf
# from keras import backend as K
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# sess = tf.Session(config=config)
# K.set_session(sess)

def load_model_(path):
    try:
        path = splitext(path)[0]
        with open('%s.json' % path, 'r') as json_file:
            model_json = json_file.read()
        model = model_from_json(model_json, custom_objects={})
        model.load_weights('%s.h5' % path)
        print("Loading model successfully...")
        return model
    except Exception as e:
        print(e)



if __name__ == "__main__":
    wpod_net_path = "wpod-net.json"
    wpod_net = load_model_(wpod_net_path)
    print(wpod_net.summary())

    # モデルオブジェクトmodelを、ディレクトリ'tfjs'に、8ビットの量子化を用いて書き出す。
    tfjs.converters.save_keras_model(wpod_net, 'tfjs', quantization_dtype=np.uint8)

