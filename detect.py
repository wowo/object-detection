#!/usr/bin/env python3.7
import logging

logging.basicConfig(level=logging.INFO, format='[%(relativeCreated)d] %(asctime)s :%(levelname)s: %(message)s')
logging.info('Starting')

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
import smtplib
import tensorflow as tf
import time

logging.info('Imports done')

tf.compat.v1.enable_eager_execution()
PUSHBULLET_API_KEY = os.environ.get('PUSHBULLET_API_KEY', None)
EMAIL_CREDENTIALS = os.environ.get('EMAIL_CREDENTIALS', None)
EMAIL_SERVER = os.environ.get('EMAIL_SERVER', 'smtp.gmail.com:587')
EMAIL_SENDER = 'mailer@sznapka.pl'
EMAIL_RECIPIENT = 'wojtek@sznapka.pl'
CAMERA_IMAGE = 'https://pihome.sznapka.pl/camera/auto.jpg'
# CAMERA_IMAGE = 'https://sznapka.pl/detection/camera-original-20210322_1820.jpg'

EXCLUDED = ['train', 'umbrella', 'kite', 'boat', 'zebra', 'clock', 'sink', 'bird', 'airplane', 'bus']
ROOT_PATH = '/var/www/sznapka.pl/detection/'
NOTIFICATION_HOST = 'https://sznapka.pl/'
NIGHT_HOURS = range(5, 21)
THRESHOLD = .5
MAX_AREA = .10


def send_email(title: str,  body: str, img: str, latest_notification: datetime) -> datetime:
    if latest_notification and datetime.now() - timedelta(seconds=60) < latest_notification:
        return latest_notification
    message = MIMEMultipart('mixed')
    message['From'] = 'Kamera <{}>'.format(EMAIL_SENDER)
    message['To'] = EMAIL_RECIPIENT
    message['Subject'] = title
    body = MIMEText(body, 'html')
    message.attach(body)

    with open(img, 'rb') as img:
        p = MIMEApplication(img.read(), _subtype='jpg')
        p.add_header('Content-Disposition', 'attachment; filename=camera.jpg')
        message.attach(p)

    server, port = EMAIL_SERVER.split(':')
    user, passwd = EMAIL_CREDENTIALS.split(':')
    with smtplib.SMTP(server, port) as server:
        server.starttls()
        server.login(user, passwd)
        server.sendmail(EMAIL_SENDER, EMAIL_RECIPIENT, message.as_string())
        server.quit()

    return datetime.now()


def send_notification(title: str,  body: str, link: str, api_key: str, latest_notification: datetime) -> datetime:
    if latest_notification and datetime.now() - timedelta(seconds=60) < latest_notification:
        return latest_notification

    data = {
        'title': title,
        'body': body,
        'type': 'link',
        'url': link
    }
    headers = {'Access-Token': api_key}
    req = requests.post('https://api.pushbullet.com/v2/pushes', data=data, headers=headers)
    logging.info('Sent notification with {}: {}'.format(link, req.status_code))
    return datetime.now()


def get_area_percentage(orig_width: float, orig_height: float, box: list) -> float:
    box_area = (orig_height * box[0] - orig_height * box[2]) * (orig_width * box[1] - orig_width * box[3])
    return box_area / (orig_height * orig_width)


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
    outpath = 'camera-detected-{}.jpg'.format(datetime.now().strftime('%Y%m%d_%H%M%S'))

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
                    logging.error('Box area is {:.2f}% for {} - ignoring'.format(area * 100, class_name))
                    scores[0][idx] = 0
                    continue
                else:
                    current_classes.append((class_name, val, area))
                    logging.warning(
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
            detection_msg = ', '.join(map(lambda x: "{}: {:.0f}% area: {:.1}%".format(x[0], x[1] * 100, x[2] * 100), current_classes))
            logging.info('Wrote to {0}'.format(outpath))
            #if PUSHBULLET_API_KEY:
            #    notification = send_notification('Detected: {}'.format(detection_msg), datetime.now().strftime('%T'),
            #                                     NOTIFICATION_HOST + outpath, PUSHBULLET_API_KEY, notification)
            if EMAIL_CREDENTIALS:
                notification = send_email('Detected: {}'.format(detection_msg), datetime.now().strftime('%T'),
                                          ROOT_PATH + outpath, notification)

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
