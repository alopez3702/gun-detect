# coding: utf-8
########################################################################################
#### Ctrl-f '#!!!' to find lines that may need to be changed, depending on host.    ####
#### As is, this script runs on the container built by the Dockerfile in this repo. ####
########################################################################################
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import pandas as pd
import cv2
import glob
import timeit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import numpy as np

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image as img
from IPython.display import Image, display, clear_output
from elasticsearch import Elasticsearch
from datetime import datetime
from minio import Minio
from minio.error import (ResponseError, BucketAlreadyOwnedByYou, BucketAlreadyExists, NoSuchKey)


# Hold warnings
import warnings
warnings.filterwarnings('ignore')



#######################################################################
################ Initialize Functions/Variables #######################
#######################################################################

#Gun type to detect
PISTOL = "pistol"
LONGGUN = "longgun"
BOTH = "both"

# Gun Type selection
gun = globals()[str((sys.argv)[1])]

#camera selection
url = sys.argv[2]
urlname = camera[-5:]

client = sys.argv[3]
access = sys.argv[4]
secret = sys.argv[5]

# Science Thresholds
person_threshold = 0.50
person_gun_threshold = 0.60

# paths to model and labels
if(gun=="longgun"):
    model = '/tf_files/retrained_graph_long_gun.pb' #!!!
    labels = "/tf_files/retrained_labels_long_gun.txt" #!!!
elif(gun=="both"):
    model = '/tf_files/frozen_inference_graph.pb'
    labels = "/tf_files/frozen_inference_graph.txt"
else:
    model = '/tf_files/retrained_graph.pb' #!!!
    labels = "/tf_files/retrained_labels.txt" #!!!

# Intialize Tensorflow session and gpu memory management
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
#config.gpu_options.per_process_gpu_memory_fraction = gpuAmount
session = tf.Session(config=config)

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
os.chdir("/tensorflow/models/research/object_detection/") #!!!

# ## Object detection imports
# Here are the imports from the object detection module.


# Needed if you want to make bounded boxes around person for object detection
from utils import label_map_util
#from utils import visualization_utils as vis_util

##################### Model Preparation ###############################

# ## Variables
#
# Any model exported using the `export_inference_graph.py` tool can be loaded here simply by changing `PATH_TO_CKPT` to point to a new .pb file.
# By default we use an "SSD with Mobilenet" model here. See the [detection model zoo](https://github.com/tensorflow/models/blob/master/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.


# What model to download.
# What model to download.
MODEL_NAME ='faster_rcnn_resnet101_coco_2017_11_08'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
NUM_CLASSES = 1

#Load a Object Detection Model(frozen) Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Object Recognition model
label_lines = [line.rstrip() for line in tf.gfile.GFile(labels)] #!!!


def initialSetup():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    start_time = timeit.default_timer()

    # This takes 2-5 seconds to run
    # Unpersists graph from file
    with tf.gfile.FastGFile(model, 'rb') as h: #!!!
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(h.read())
        tf.import_graph_def(graph_def, name='')

    print('Took {} seconds to unpersist the graph'.format(timeit.default_timer() - start_time))


initialSetup()


# ## Helper code for frame processing
def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)

minioClient = Minio(client,
                  access_key=access,
                  secret_key=secret,
                  secure=True)

try: 
    minioClient.make_bucket("test", location="us-east-1")
except BucketAlreadyOwnedByYou as err:
    pass
except BucketAlreadyExists as err:
    pass
except ResponseError as err:
    raise

count = 0
person_count = 0
location = '/pistol-detection/detect_pistol/Data' #!!!
os.chdir(location)
with tf.Session() as sess2:
    while(True):
        image = 'rec_frame'+urlname+'_'+str(count)+'.jpg'
        csv = 'rec_frame'+urlname+'_'+str(count)+'.csv'
        try:
            minioClient.fget_object('test', image, image)
            minioClient.fget_object('test', csv, csv)
        except NoSuchKey as err:
            #Note: Session 2 may go through files on the Minio server faster than Session 1 can produce files
            #This error may appear frequently if Session 2 can't find a file Session 1 hadn't produced yet
            #Session 2 will wait for Session 1 to make the next file.
            print(err) #comment this out if you don't want constant error messages
            print("NoSuchKey error. Tried to find a non-existent file in the Minio Client") #Comment this one out too
            continue
        except ResponseError as err:
            print(err)
            continue
        image_np = cv2.imread(image)
        df8 = pd.read_csv(csv)
        
        #If the Minio Client bucket gets too full of images, use the following code to delete files.
        #We choose to not delete files because if files are deleted and Session 2 is restarted but Session 1 is not,
        #Session 2 will be stuck in an infinite loop trying to find files that it already deleted.
        #But for live-footage, Session 1 and Session 2 should be restarted together if they need to be.
        #This code can also be used for "garbage collection".
        #try:
        #    minioClient.remove_object('test', image)
        #    minioClient.remove_object('test', csv)
        #except ResponseError as err:
        #    print(err)
        
        count+=1
        
        wid = df8['wid'].tolist()
        hei = df8['hei'].tolist()
        px = df8['px'].tolist()
        py = df8['py'].tolist()
        
        start_time = timeit.default_timer()
        for person in range(0,len(df8.index) - 1):
            softmax_tensor = sess2.graph.get_tensor_by_name('final_result:0')

            upperbody = hei[person] / 3
            roi = image_np[int(py[person]):int(py[person]) + int(upperbody), int(px[person]):int(px[person]) + int(wid[person])]

            frame = cv2.resize(roi, (299, 299), interpolation=cv2.INTER_CUBIC)

                    # adhere to TS graph input structure
            numpy_frame = np.asarray(frame)
            numpy_frame = cv2.normalize(numpy_frame.astype('float'), None, -0.5, .5, cv2.NORM_MINMAX)
            numpy_final = np.expand_dims(numpy_frame, axis=0)


                    # make prediciton
            predictions = sess2.run(softmax_tensor, {'Mul:0': numpy_final})

            score = predictions.item(1)
            gunScore = str(score)
                    
#                     ipx = []
#                     ipy = []
#                     iwid = []
#                     ihei = []

                    # Add Red Box to image under conditions
            if score > person_gun_threshold:
                person_count += 1
                cv2.rectangle(image_np, (px[person],py[person]), (px[person]+wid[person], py[person]+ hei[person]), (0, 0, 255), 10)
                #print(px[person],py[person],wid[person],hei[person])
                        #ipx.append()
                labelBuffer = int(py[person]) - int(hei[person] * 0.1)

                        # print
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(image_np, gunScore, (int(px[person]), labelBuffer), font, 0.8, (0, 255, 0), 2)

        print('Took {} seconds to perform image recognition on people found'.format(timeit.default_timer()))
        cv2.imshow('frame', image_np)
        if cv2.waitKey(1000) & 0xFF == ord('q'):
            os.remove(image)
            os.remove(csv)
            break
        
        os.remove(image)
        os.remove(csv)
sess2.close()
cv2.destroyWindow('frame')
