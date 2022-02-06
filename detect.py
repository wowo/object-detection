#!/usr/bin/env python3.7
import logging

logging.basicConfig(level=logging.INFO, format='[%(relativeCreated)d] %(asctime)s :%(levelname)s: %(message)s')
logging.info('Starting')

from detect_utils import box_contains_exluded_points, send_email, get_area_percentage

from PIL import Image
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
from utils import label_map_util
from utils import visualization_utils as vis_util
import cv2
import numpy as np
import os
import requests
import tensorflow as tf
import time

logging.info('Imports done')

tf.compat.v1.enable_eager_execution()
EMAIL_CREDENTIALS = os.environ.get('EMAIL_CREDENTIALS', None)
EMAIL_SERVER = os.environ.get('EMAIL_SERVER', 'smtp.gmail.com:587')
EMAIL_SENDER = 'mailer@sznapka.pl'
EMAIL_RECIPIENT = 'wojtek@sznapka.pl,piotr@sznapka.pl'
CAMERA_IMAGE = 'https://pihome.sznapka.pl/camera/auto.jpg'
#CAMERA_IMAGE = 'https://sznapka.pl/detection/camera-detected-20220206_1146.jpg'

EXCLUDED = ['train', 'umbrella', 'kite', 'boat', 'zebra', 'clock', 'sink', 'bird', 'airplane', 'bus', 'giraffe', 'traffic light']
EXCLUDED_POINTS = [[710,180]] # wykluczone punkty środkowe, np załamanie bramy Piora, które jest klasyfikowane jako auto
ROOT_PATH = '/var/www/sznapka.pl/detection/'
NOTIFICATION_HOST = 'https://sznapka.pl/'
NIGHT_HOURS = range(5, 23)
THRESHOLD = .6
MAX_AREA = .1


MODEL_NAME = 'ssdlite_mobilenet_v2_coco_2018_05_09'
CWD_PATH = os.getcwd()
PATH_TO_GRAPH = os.path.join(CWD_PATH, MODEL_NAME, 'frozen_inference_graph.pb')
PATH_TO_LABELS = os.path.join(CWD_PATH, 'models/research/object_detection/data', 'mscoco_label_map.pbtxt')
NUM_CLASSES = 90
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(PATH_TO_GRAPH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
    sess = tf.compat.v1.Session(graph=detection_graph)

image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

logging.info('All initialization done')
notification = None

while True:
    # for path in glob.glob('/tmp/w*'):
    path = '/tmp/camera.jpg'
    outpath = 'camera-detected-{}.jpg'.format(datetime.now().strftime('%Y%m%d_%H%M'))

    try:
        r = requests.get(CAMERA_IMAGE)
        if r.status_code != 200:
            raise Exception("Error {0}: {1}".format(r.status_code, r.content))
        with open(path, 'wb') as f:
            f.write(r.content)

        pil_image = Image.open(path)
        width, height = pil_image.size
        image = np.array(pil_image)
        input_tensor = tf.convert_to_tensor(image)
        input_tensor = input_tensor[tf.newaxis, ...]

        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: input_tensor.numpy()})

        current_classes = []
        for idx, val in enumerate(scores[0]):
            class_name = category_index[classes[0][idx]]['name']
            if val > THRESHOLD:
                area = get_area_percentage(width, height, boxes[0][idx])
                if class_name in EXCLUDED:
                    scores[0][idx] = 0
                    continue
                elif area > MAX_AREA:
                    logging.warning('Box area is {:.2f}% for {} - ignoring'.format(area * 100, class_name))
                    scores[0][idx] = 0
                    continue
                elif box_contains_exluded_points(width, height, boxes[0][idx], EXCLUDED_POINTS):
                    logging.warning('Box {} contains one of excluded points {} - ignoring'.format(boxes[0][idx], EXCLUDED_POINTS))
                    scores[0][idx] = 0
                    continue
                else:
                    current_classes.append((class_name, val, area, boxes[0][idx]))
                    logging.info(
                        'Found at index {} class: {} with {:.0f}% boxes: {}'.format(idx, class_name, val * 100,
                                                                                    boxes[0][idx]))

        if len(current_classes) > 0:
            cv2.imwrite(ROOT_PATH + outpath.replace('detected', 'original'), image)
            vis_util.visualize_boxes_and_labels_on_image_array(
                image,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=1,
                min_score_thresh=THRESHOLD)
            cv2.imwrite(ROOT_PATH + outpath, image)
            print(current_classes)
            detection_title  = 'Detected: ' + ', '.join(map(lambda x: x[0], current_classes))
            detection_msg = ', '.join(map(lambda x: "{}: {:.0f}% area: {:.1}%, box: {}".format(x[0], x[1] * 100, x[2] * 100, x[3]), current_classes))
            logging.info('Wrote to {0}'.format(outpath))
            if EMAIL_CREDENTIALS:
                notification = send_email(EMAIL_SENDER, EMAIL_RECIPIENT, EMAIL_SERVER, EMAIL_CREDENTIALS,
                                          detection_title, detection_msg, ROOT_PATH + outpath, notification)

        hour = int(datetime.now().strftime('%H'))
        if hour not in NIGHT_HOURS:
            logging.info('It is {} - sleeping for 10 minutes'.format(datetime.now().strftime('%H:%M')))
            time.sleep(60 * 10)
    except Exception as err:
        exception_type = type(err).__name__
        msg = "{} occurred {}".format(exception_type, str(err))
        if isinstance(err, ValueError) and 'Unsupported object type JpegImageFile' in str(err):
            logging.debug(msg)
        else:
            logging.error(msg)
