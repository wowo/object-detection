#!/usr/bin/env python3.7
import logging

logging.basicConfig(level=logging.INFO, format='[%(relativeCreated)d] %(asctime)s :%(levelname)s: %(message)s')
logging.info('Starting')

from PIL import Image
from datetime import datetime, timedelta
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
PUSHBULLET_API_KEY = os.environ.get('PUSHBULLET_API_KEY', None)
CAMERA_IMAGE = 'https://pihome.sznapka.pl/camera/auto.jpg'


def send_notification(title, body, link, api_key, latest_notification):
    if latest_notification and datetime.now() - timedelta(seconds=60) < latest_notification:
        return latest_notification

    data = {
        'title': title,
        'body': body,
        'type': 'link',
        'url': link
    }
    headers = {'Access-Token': api_key}
    requests.post('https://api.pushbullet.com/v2/pushes', data=data, headers=headers)
    logging.info('Sent notification with {0}'.format(link))
    return datetime.now()


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

excluded_classes = ['train', 'umbrella', 'kite', 'boat', 'zebra', 'clock']
rootpath = '/var/www/sznapka.pl/'
notification_host = 'https://sznapka.pl/'
night_hours = range(5, 21)
THRESHOLD = .5

while True:
    # for path in glob.glob('/tmp/w*'):
    path = '/tmp/camera.jpg'
    logging.info('Fetching image to {}'.format(path))
    outpath = 'detection/camera-detected-{}.jpg'.format(datetime.now().strftime('%Y%m%d_%H%M'))

    try:
        r = requests.get(CAMERA_IMAGE)
        if r.status_code != 200:
            raise "Error {0}: {1}".format(r.status_code, r.content)
        with open(path, 'wb') as f:
            f.write(r.content)

        image = np.array(Image.open(path))
        input_tensor = tf.convert_to_tensor(image)
        input_tensor = input_tensor[tf.newaxis, ...]

        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: input_tensor.numpy()})

        current_classes = []
        for idx, val in enumerate(scores[0]):
            class_name = category_index[classes[0][idx]]['name']
            if val > THRESHOLD:
                if class_name in excluded_classes:
                    scores[0][idx] = 0
                    continue
                else:
                    current_classes.append((class_name, val))
                    logging.warning(
                        'Found at index {} class: {} with {:.0f}% boxes: {}'.format(idx, class_name, val * 100,
                                                                                    boxes[0][idx]))

        if len(current_classes) > 0:
            cv2.imwrite(rootpath + outpath.replace('detected', 'original'), image)
            vis_util.visualize_boxes_and_labels_on_image_array(
                image,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=1,
                min_score_thresh=THRESHOLD)
            cv2.imwrite(rootpath + outpath, image)
            detection_msg = ', '.join(map(lambda x: "{}: {:.0f}%".format(x[0], x[1] * 100), current_classes))
            logging.info('Wrote to {0}'.format(outpath))
            if PUSHBULLET_API_KEY:
                notification = send_notification('Detected: {}'.format(detection_msg), datetime.now().strftime('%T'),
                                                 notification_host + outpath, PUSHBULLET_API_KEY, notification)

        hour = int(datetime.now().strftime('%H'))
        if hour not in night_hours:
            logging.info('It is {} - sleeping for 10 minutes'.format(datetime.now().strftime('%H:%M')))
            time.sleep(60 * 10)
    except Exception as err:
        exception_type = type(err).__name__
        logging.error("{} occured {}".format(exception_type, str(err)))
