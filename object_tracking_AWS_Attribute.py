import cv2
import torch
import numpy as np
from absl import app, flags
from absl.flags import FLAGS
from deep_sort_realtime.deepsort_tracker import DeepSort
from models.common import DetectMultiBackend, AutoShape

import time
import boto3
import psutil

from botocore.config import Config

import os
import json
import torch
import argparse
from PIL import Image
from torchvision import transforms as T
from net import get_model

#AWS region setting
REGION = "ap-northeast-1"

#Timestream database table name settings
DATABASE_NAME = "sampleDB"
TABLE_NAME = "SampleTable"

#Dimension name settings
COUNTRY = "Japan"
CITY = "Tokyo"
HOSTNAME = "DemoInstance01"

# Define command line flags
flags.DEFINE_string('video', './data/test.mp4', 'Path to input video or webcam index (0)')
flags.DEFINE_string('output', './output/output.mp4', 'path to output video')
flags.DEFINE_float('conf', 0.50, 'confidence threshold')
flags.DEFINE_integer('blur_id', None, 'class ID to apply Gaussian Blur')
flags.DEFINE_integer('class_id', None, 'class ID to track')

######################################################################
# Settings
# ---------
dataset_dict = {
    'market'  :  'Market-1501',
    'duke'  :  'DukeMTMC-reID',
}
num_cls_dict = { 'market':30, 'duke':23 }
num_ids_dict = { 'market':751, 'duke':702 }

transforms = T.Compose([
    T.Resize(size=(288, 144)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

######################################################################
# Argument
# ---------
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='market', type=str, help='dataset')
parser.add_argument('--backbone', default='resnet50', type=str, help='model')
parser.add_argument('--use-id', action='store_true', help='use identity loss')
args = parser.parse_args()

assert args.dataset in ['market', 'duke']
assert args.backbone in ['resnet50', 'resnet34', 'resnet18', 'densenet121']

model_name = '{}_nfc_id'.format(args.backbone) if args.use_id else '{}_nfc'.format(args.backbone)
num_label, num_id = num_cls_dict[args.dataset], num_ids_dict[args.dataset]


######################################################################
# Model and Data
# ---------
def load_network(network):
    save_path = os.path.join('./checkpoints', "market", model_name, 'net_last.pth')
    network.load_state_dict(torch.load(save_path))
    print('Resume model from {}'.format(save_path))
    return network

def load_image(path):
    src = Image.open(path)
    src = transforms(src)
    src = src.unsqueeze(dim=0)
    return src

def transform_image(src):
    src = transforms(src)
    src = src.unsqueeze(dim=0)
    return src

model_attr = get_model(model_name, num_label, use_id=args.use_id, num_id=num_id)
model_attr = load_network(model_attr)
model_attr.eval()

######################################################################
# Inference
# ---------
class predict_decoder(object):

    def __init__(self, dataset):
        with open('./doc/label.json', 'r') as f:
            self.label_list = json.load(f)[dataset]
        with open('./doc/attribute.json', 'r') as f:
            self.attribute_dict = json.load(f)[dataset]
        self.dataset = dataset
        self.num_label = len(self.label_list)

    def decode(self, pred):
        pred = pred.squeeze(dim=0)
        for idx in range(self.num_label):
            name, chooce = self.attribute_dict[self.label_list[idx]]
            if chooce[pred[idx]]:
                print('{}: {}'.format(name, chooce[pred[idx]]))



def main(_argv):

    #Functions that set dimensions
    def prepare_dimensions():
        dimensions = [
                {'Name': 'country', 'Value': COUNTRY},
                {'Name': 'city', 'Value': CITY},
                {'Name': 'hostname', 'Value': HOSTNAME}
        ]
        return dimensions

    #Function that writes data to timestream
    def write_records(records):
        try:
            result = write_client.write_records(DatabaseName=DATABASE_NAME,
                                                TableName=TABLE_NAME,
                                                CommonAttributes={},
                                                Records=records)
            status = result['ResponseMetadata']['HTTPStatusCode']
            print("WriteRecords HTTPStatusCode: %s" %
                (status))
        except Exception as err:
            print("Error:", err)

    # Initialize the video capture
    # video_input = FLAGS.video
    video_input = 0
    # Check if the video input is an integer (webcam index)
    if FLAGS.video.isdigit():
        video_input = int(video_input)
        cap = cv2.VideoCapture(video_input)
    else:
        cap = cv2.VideoCapture(video_input)
    if not cap.isOpened():
        print('Error: Unable to open video source.')
        return
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    # video writer objects
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(FLAGS.output, fourcc, fps, (frame_width, frame_height))

    # Initialize the DeepSort tracker
    tracker = DeepSort(max_age=50)
    # select device (CPU or GPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load YOLO model
    model = DetectMultiBackend(weights='./weights/yolov9-e.pt',device=device, fuse=True)
    model = AutoShape(model)

    # Load the COCO class labels
    classes_path = "../configs/coco.names"
    with open(classes_path, "r") as f:
        class_names = f.read().strip().split("\n")

    # Create a list of random colors to represent each class
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(len(class_names), 3)) 

    print("writing data to database {} table {}".format(
    DATABASE_NAME, TABLE_NAME))

    session = boto3.Session(region_name=REGION)
    write_client = session.client('timestream-write',
                                config=Config(read_timeout=20,
                                max_pool_connections=5000,
                                retries={'max_attempts': 10}))
    dimensions = prepare_dimensions()

    records = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Run model on each frame
        results = model(frame)
        detect = []
        for det in results.pred[0]:
            label, confidence, bbox = det[5], det[4], det[:4]
            x1, y1, x2, y2 = map(int, bbox)
            class_id = int(label)

            # Filter out weak detections by confidence threshold and class_id
            if FLAGS.class_id is None:
                if confidence < FLAGS.conf:
                    continue
            else:
                if class_id != FLAGS.class_id or confidence < FLAGS.conf:
                    continue

            detect.append([[x1, y1, x2 - x1, y2 - y1], confidence, class_id])

        tracks = tracker.update_tracks(detect, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            ltrb = track.to_ltrb()
            class_id = track.get_det_class()
            x1, y1, x2, y2 = map(int, ltrb)
            color = colors[class_id]
            B, G, R = map(int, color)
            text = f"{track_id} - {class_names[class_id]}"

            # frameから人の枠を切り取る
            crop_img = frame[y1:y2, x1:x2]
            # 画像をPIL形式に変換
            img = Image.fromarray(crop_img)
            # 画像をモデルに合わせて変換
            src = transform_image(img)

            print(src.shape, src.dtype, src.device)
            out = model_attr.forward(src)

            pred = torch.gt(out, torch.ones_like(out)/2 )  # threshold=0.5

            Dec = predict_decoder("market")
            Dec.decode(pred)

            # Write data to Timestream

            print(text)
            current_time = str(round(time.time() * 1000))
        
            cpu_utilization = {
                'Dimensions': dimensions,
                'MeasureName': 'cpu_utilization',
                'MeasureValue': str(psutil.cpu_percent()),
                'MeasureValueType': 'DOUBLE',
                'Time': current_time
            }
            
            memory_utilization = {
                'Dimensions': dimensions,
                'MeasureName': 'memory_utilization',
                'MeasureValue': str(psutil.virtual_memory().percent),
                'MeasureValueType': 'DOUBLE',
                'Time': current_time
            }
            
            
            disk_utilization = {
                'Dimensions': dimensions,
                'MeasureName': 'disk_utilization',
                'MeasureValue': str(psutil.disk_usage('/').percent),
                'MeasureValueType': 'DOUBLE',
                'Time': current_time
            }

            detected_people = {
                'Dimensions': dimensions,
                'MeasureName': 'detected_people',
                'MeasureValue': text,
                'MeasureValueType': 'VARCHAR',
                'Time': current_time
            }


            records = [cpu_utilization, memory_utilization, disk_utilization, detected_people]
            write_records(records)
            records = []


            # Visualize
            cv2.rectangle(frame, (x1, y1), (x2, y2), (B, G, R), 2)
            cv2.rectangle(frame, (x1 - 1, y1 - 20), (x1 + len(text) * 12, y1), (B, G, R), -1)
            cv2.putText(frame, text, (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Apply Gaussian Blur
            if FLAGS.blur_id is not None and class_id == FLAGS.blur_id:
                if 0 <= x1 < x2 <= frame.shape[1] and 0 <= y1 < y2 <= frame.shape[0]:
                    frame[y1:y2, x1:x2] = cv2.GaussianBlur(frame[y1:y2, x1:x2], (99, 99), 3)



        cv2.imshow('YOLOv9 Object tracking', frame)
        writer.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release video capture and writer
    cap.release()
    writer.release()

if __name__ == '__main__':
  try:
      app.run(main)
  except SystemExit:
      pass
