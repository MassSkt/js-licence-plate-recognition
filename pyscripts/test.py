import cv2
import numpy as np
import matplotlib.pyplot as plt
from local_utils import detect_lp
from os.path import splitext,basename
from keras.models import model_from_json,load_model
import glob
import time


import tensorflow as tf
from keras import backend as K
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)

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

def get_plate(image_path, Dmax=608, Dmin=256):
    vehicle = preprocess_image(image_path)
    ratio = float(max(vehicle.shape[:2])) / min(vehicle.shape[:2])
    side = int(ratio * Dmin)
    bound_dim = min(side, Dmax)
    _ , LpImg, _, cor = detect_lp(wpod_net, vehicle, bound_dim, lp_threshold=0.5)
    return LpImg, cor

def preprocess_image(image_path,resize=False):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255
    if resize:
        img = cv2.resize(img, (224,224))
    return img

def draw_box(image_path, cor, thickness=3): 
    pts=[]  
    x_coordinates=cor[0][0]
    y_coordinates=cor[0][1]
    # store the top-left, top-right, bottom-left, bottom-right 
    # of the plate license respectively
    for i in range(4):
        pts.append([int(x_coordinates[i]),int(y_coordinates[i])])
    
    pts = np.array(pts, np.int32)
    pts = pts.reshape((-1,1,2))
    vehicle_image = preprocess_image(image_path)
    
    cv2.polylines(vehicle_image,[pts],True,(0,255,0),thickness)
    return vehicle_image


if __name__ == "__main__":
    wpod_net_path = "pyscripts/wpod-net.json"
    wpod_net = load_model_(wpod_net_path)
    print(wpod_net.summary())


    # Create a list of image paths 
    image_paths = glob.glob("test_img/*")
    print("Found %i images..."%(len(image_paths)))

    # Obtain plate image and its coordinates from an image
    test_image = image_paths[0]
    st=time.time()
    LpImg,cor = get_plate(test_image)
    print("elapsed",time.time()-st)
    print("Detect %i plate(s) in"%len(LpImg),splitext(basename(test_image))[0])
    print("Coordinate of plate(s) in image: \n", cor)

    # Visualize our result
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.axis(False)
    plt.imshow(preprocess_image(test_image))
    plt.subplot(1,2,2)
    plt.axis(False)
    plt.imshow(LpImg[0])

    plt.figure(figsize=(8,8))
    plt.axis(False)
    plt.imshow(draw_box(test_image,cor))

    plt.show()
    plt.close()
