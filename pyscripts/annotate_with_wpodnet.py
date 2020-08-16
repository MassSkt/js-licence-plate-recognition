import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Or 2, 3, etc. other than 0

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config = tf.ConfigProto(
    gpu_options=tf.GPUOptions(
        visible_device_list="0", # specify GPU number
        allow_growth=True
    )
)
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras


import cv2
import numpy as np
import matplotlib.pyplot as plt
from local_utils import detect_lp
from os.path import splitext,basename
from keras.models import model_from_json,load_model
import glob
import time
import os


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

def get_plate(image_path, Dmax=608, Dmin=256,lp_threshold=0.5):
    vehicle = preprocess_image(image_path)
    ratio = float(max(vehicle.shape[:2])) / min(vehicle.shape[:2])
    side = int(ratio * Dmin)
    bound_dim = min(side, Dmax)
    _ , LpImg, _, cor = detect_lp(wpod_net, vehicle, bound_dim, lp_threshold)
    return LpImg, cor

def preprocess_image(image_path,resize=False):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255
    if resize:
        img = cv2.resize(img, (224,224))
    return img

def draw_box(image_path, cor, thickness=3): 
    vehicle_image = cv2.imread(image_path)
    vehicle_image = cv2.cvtColor(vehicle_image, cv2.COLOR_BGR2RGB)

    for k in range(len(cor)):
        pts=[]  
        x_coordinates=cor[k][0]
        y_coordinates=cor[k][1]
        # store the top-left, top-right, bottom-left, bottom-right 
        # of the plate license respectively
        for i in range(4):
            pts.append([int(x_coordinates[i]),int(y_coordinates[i])])
        
        pts = np.array(pts, np.int32)
        pts = pts.reshape((-1,1,2))
        print(pts)
    #     vehicle_image = preprocess_image(image_path)
        
        cv2.polylines(vehicle_image,[pts],True,(0,255,0),thickness)
    return vehicle_image

if __name__ == "__main__":
    wpod_net_path = "pyscripts/wpod-net.json"
    wpod_net = load_model_(wpod_net_path)
    print(wpod_net.summary())


    # Create a list of image paths 
    image_paths = glob.glob("train_data/self_taken/*")
    print("Found %i images..."%(len(image_paths)))
    base_save_dir="./train_data/processed"
    os.makedirs(base_save_dir,exist_ok=True)

    for test_image in image_paths:
        # Obtain plate image and its coordinates from an image
        print("-"*40)
        print(test_image)
        image = cv2.imread(test_image)
        try:
            LpImg,cor = get_plate(test_image,lp_threshold=0.5)
        except Exception as e:
            print(e)
            continue
        labeled_image=draw_box(test_image,cor)
        x_coordinates=cor[0][0]
        y_coordinates=cor[0][1]
        print("Detect %i plate(s) in"%len(LpImg),splitext(basename(test_image))[0])
        print("x Coordinate:",x_coordinates)
        print("y Coordinate:",y_coordinates)
        base_name=splitext(basename(test_image))[0]
        ext=test_image.split(".")[-1]

        #save
        print(os.path.join(base_save_dir,base_name+"."+ext))
        cv2.imwrite(os.path.join(base_save_dir,base_name+"."+ext),image)
        cv2.imwrite(os.path.join(base_save_dir,base_name+"_detect."+ext),labeled_image[:,:,::-1])
        
        # save y (coordinate 1st row x , 2nd row y) as txt
        path_w = os.path.join(base_save_dir,base_name+".txt")
        with open(path_w, mode='w') as f:
            f.write(','.join(x_coordinates.astype(np.int).astype(str).tolist()))
            f.write('\n')
            f.write(','.join(y_coordinates.astype(np.int).astype(str).tolist()))
    # Visualize our result
    # plt.figure(figsize=(12,5))
    # plt.subplot(1,2,1)
    # plt.axis(False)
    # plt.imshow(preprocess_image(test_image))
    # plt.subplot(1,2,2)
    # plt.axis(False)
    # plt.imshow(LpImg[0])

    # plt.figure(figsize=(8,8))
    # plt.axis(False)
    # plt.imshow(draw_box(test_image,cor))

    # plt.show()
    # plt.close()
