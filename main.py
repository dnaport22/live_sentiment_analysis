print("Importing modules...")
import numpy as np
import tensorflow as tf
import os
import time
import six.moves.urllib as urllib
import tarfile

# Imports the Google Cloud client library
import cv2
from core.detection_graph import DetectionGraph
from helper.object_slicer import ObjectSlicer
from utils import label_map_util
from helper.sentiment_analysis import SentimentAnalysis

print("Connecting to camera...")
cap = cv2.VideoCapture(0)
time.sleep(3)

print("Updating object detection model...")
MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

opener = urllib.request.URLopener()
opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
tar_file = tarfile.open(MODEL_FILE)
for file in tar_file.getmembers():
  file_name = os.path.basename(file.name)
  if 'frozen_inference_graph.pb' in file_name:
      tar_file.extract(file, os.getcwd())

print("Initialising program...")
SHOW_IMAGE_OUTPUT = False

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90

graph = DetectionGraph()
detection_graph = graph.get_model()

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)

print("Creating Tensorflow session...")
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

    print("Running model...")
    while True:
        ret, image_np = cap.read()
        # image = Image.open(image_path)
        # the array based representation of the image will be used later in order to prepare the
        # result image with boxes and labels on it.
        # image_np = load_image_into_numpy_array(image)
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        # Actual detection.
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})
        # Visualization of the results of a detection.
        slicer = ObjectSlicer(
            image_np,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=8
        )
        slicer.slice_object()
        analysis = SentimentAnalysis(slicer.image_to_analyse)
        print(analysis.get_sentiments())

        if SHOW_IMAGE_OUTPUT:
            cv2.imshow('object detection', cv2.resize(image_np, (1220, 600)))
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
            break




