import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import cv2 as cv

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

## This is needed to display the images.
#%matplotlib inline

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

from utils import label_map_util

from utils import visualization_utils as vis_util

# What model to download.
#MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
#MODEL_FILE = MODEL_NAME + '.tar.gz'
#DOWNLOAD_BASE = #'http://download.tensorflow.org/models/object_detection/'
MODEL_NAME = './output/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = './dataset_label_map.pbtxt'

NUM_CLASSES = 1

#download model
#opener = urllib.request.URLopener()
#opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
#tar_file = tarfile.open(MODEL_FILE)
#for file in tar_file.getmembers():
#  file_name = os.path.basename(file.name)
#  if 'frozen_inference_graph.pb' in file_name:
#    tar_file.extract(file, os.getcwd())

#Load a (frozen) Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')
#Loading label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)
#Helper code
def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


tf.app.flags.DEFINE_string('data','None','the path of the txt')
tf.app.flags.DEFINE_string('image','None','the path of the txt')
FLAGS = tf.app.flags.FLAGS

TEST_IMAGE = FLAGS.image
IMAGE_TEST = FLAGS.data

if (TEST_IMAGE=='None' and IMAGE_TEST == 'None'):
      print("----------------------------------------------------------------------------\n")
      print("please use --data to point the val.txt or use --image point the single image!\n")
      os._exit(0)

with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    # Definite input and output Tensors for detection_graph
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    # Each box represents a part of the image where a particular object was detected.
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    
    if TEST_IMAGE != 'None':
        #for image_path in TEST_IMAGE_PATHS:
        image = Image.open(TEST_IMAGE)
        # the array based representation of the image will be used later in order to prepare the
        # result image with boxes and labels on it.
        image_np = load_image_into_numpy_array(image)
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
    
    
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    # Each box represents a part of the image where a particular object was detected.
    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    scores = detection_graph.get_tensor_by_name('detection_scores:0')
    classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    
    
    if IMAGE_TEST != 'None':
        f = open(IMAGE_TEST)
        Tests = f.readlines()
    
        for line in Tests:
            line = line.strip('\n')+'.jpg'
            print("**********************",line)
            imgpath = './VOCdevkit/VOC2007/JPEGImages/'+line 
            img = cv.imread(imgpath)
            img = cv.cvtColor(img,cv.COLOR_BGR2RGB)
            img_extend = np.expand_dims(img, axis=0)
    
            # Actual detection.
            (boxes_, scores_, classes_, num_detections_) = sess.run(
            [boxes, scores, classes, num_detections],
            feed_dict={image_tensor: img_extend})
            # Visualization of the results of a detection.
            vis_util.visualize_boxes_and_labels_on_image_array(
                img,
                np.squeeze(boxes_),
                np.squeeze(classes_).astype(np.int32),
                np.squeeze(scores_),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=8)

            print(scores_)  
            print(classes_)  
            print(category_index) 
   
            final_score = np.squeeze(scores_)    
            count = 0
            for i in range(100):
                if scores_ is None or final_score[i] > 0.5:
                    count = count + 1
            print('the count of objects is: ', count)   

            img = cv.cvtColor(img,cv.COLOR_RGB2BGR)
            cv.imshow("ob",img)
            cv.waitKey(1500)
            cv.destroyAllWindows()
    else:
         # Actual detection.
            img = image_np
            img_extend = image_np_expanded
            (boxes_, scores_, classes_, num_detections_) = sess.run(
            [boxes, scores, classes, num_detections],
            feed_dict={image_tensor: img_extend})
            # Visualization of the results of a detection.
            vis_util.visualize_boxes_and_labels_on_image_array(
                img,
                np.squeeze(boxes_),
                np.squeeze(classes_).astype(np.int32),
                np.squeeze(scores_),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=8)

            print(scores_)  
            print(classes_)  
            print(category_index) 
   
            final_score = np.squeeze(scores_)    
            count = 0
            for i in range(100):
                if scores_ is None or final_score[i] > 0.5:
                    count = count + 1
            print('the count of objects is: ', count)   

            img = cv.cvtColor(img,cv.COLOR_RGB2BGR)
            cv.imshow("ob",img)
            cv.waitKey(1500)
            cv.destroyAllWindows()
