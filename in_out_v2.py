import os
import time
import argparse
from sort import *
import json
import numpy as np
import subprocess

import pycuda.autoinit  # This is needed for initializing CUDA driver

from utils.yolo_classes import get_cls_dict
from utils.camera import add_camera_args, Camera
from utils.display import open_window, set_display, show_fps
from utils.visualization import BBoxVisualization
from utils.yolo_with_plugins import TrtYOLO
from azure.iot.device import IoTHubDeviceClient, Message
from datetime import datetime
from imutils.video import FPS
from datetime import timedelta
import time
tracker = Sort()
memory = {}
import cv2

def logs(TEXT_TO_SEND):
    file_object = open('/home/logs.txt', 'a')
    file_object.write((TEXT_TO_SEND+"\n"))
    # Close the file
    file_object.close()

def send_to_iot_hub(client,text):

    message = Message(text)
    message.content_encoding = "utf-8"
    message.content_type = "application/json"
    try: client.send_message(message)
    except : logs(text)
    print ("Message '{}' successfully sent".format(text))

WINDOW_NAME = 'TrtYOLODemo'

def logs(TEXT_TO_SEND):
    file_object = open('/home/logs.txt', 'a')
    # Append 'hello' at the end of file
    TEXT_TO_SEND='{'+str(TEXT_TO_SEND)+'}'
    #print(TEXT_TO_SEND)
    #print(str(dict(TEXT_TO_SEND)))
    file_object.write((TEXT_TO_SEND))
    file_object.write('\n')
    # Close the file
    file_object.close()
def intersect(A,B,C,D):
	return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def ccw(A,B,C):
	return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

def parse_args():
    """Parse input arguments."""
    desc = ('Capture and display live camera video, while doing '
            'real-time object detection with TensorRT optimized '
            'YOLO model on Jetson')
    parser = argparse.ArgumentParser(description=desc)
    parser = add_camera_args(parser)
    parser.add_argument(
        '-c', '--category_num', type=int, default=80,
        help='number of object categories [80]')
    parser.add_argument(
        '-t', '--conf_thresh', type=float, default=0.3,
        help='set the detection confidence threshold')
    parser.add_argument(
        '-m', '--model', type=str,required=True,
        help=('[yolov3-tiny|yolov3|yolov3-spp|yolov4-tiny|yolov4|'
              'yolov4-csp|yolov4x-mish|yolov4-p5]-[{dimension}], where '
              '{dimension} could be either a single number (e.g. '
              '288, 416, 608) or 2 numbers, WxH (e.g. 416x256)'))
    parser.add_argument(
        '-l', '--letter_box', action='store_true',
        help='inference with letterboxed image [False]')
    args = parser.parse_args()
    return args


def loop_and_detect(cam, trt_yolo, confidence, vis):
    #cam=cv2.VideoCapture(video_source)
    with open('/home/mac.txt','r') as file:
        device_id = file.read()
        device_id=device_id.rstrip()
        device_id=device_id.lstrip()
    with open('/home/cs.txt','r') as file2:
        cs = file2.read()
        cs=cs.rstrip()
        cs=cs.lstrip()
    CONNECTION_STRING = cs 
    CLIENT = IoTHubDeviceClient.create_from_connection_string(CONNECTION_STRING)
    TEXT_TO_SEND = ""


    camera_Disconnect = 0
    frameIndex = 0
    DEVICE_ID = device_id #os.getenv("id")
    data = json.loads(open("/home/video_source.json", 'r').read())
    video_source = data['source']
    in_X1= int(data['in_X1'])
    in_Y1= int(data['in_Y1'])
    in_X2= int(data['in_X2'])
    in_Y2= int(data['in_Y2'])
    out_X1 =int(data['out_X1'])
    out_X2=int(data['out_X2'])
    out_Y1= int(data['out_Y1'])
    out_Y2= int(data['out_Y2'])
    #in1, in2, counter = (in_X1, in_Y1), (in_X2, in_X2), 0 #(10,784), (1750,784), 0
    out1,out2=(out_X1, out_Y1), (out_X2, out_Y2)
    in1= (in_X1, in_Y1)
    in2=(in_X2, in_Y2)
    
    roi_in=str(in1)+str(in2)
    roi_out=str(out1)+str(out2)
    skip_frames= int(data['skip_frames'])
    minutes_freq = int(data['minutes_freq'])
    freq = int(data['freq_iot_hub'])
    #confidence=float(data['confidence'])
    day_Confidence=float(data['day_Confidence'])
    night_Confidence=float(data['night_Confidence'])
    start_Daytime=int(data['start_Daytime'])
    end_Daytime=int(data['end_Daytime'])
    #roi=str(start_point) + str(end_point)
    cip= data['cip']

    camera_Disconnect=0
    now=datetime.now()
    now=now.strftime("%H:%M:%S")
    if int(now[1])==0 and int(now[0])==0:
        if os.path.exists('/home/logs.txt'):
            os.remove('/home/logs.txt')
        time.sleep(60)
        subprocess.run(["sudo","reboot"])

    full_scrn = False
    fps = 0.0
    tic = time.time()
    indexIDs = []
    c = []
    memory = {}
    previous = memory.copy()
    counter=0
    counter1=0
    counter2=0
    #start_point=(405,263)
    #end_point=(1859,284)
    (W, H) = (None, None)
    to_compare = datetime.now() + timedelta(minutes=minutes_freq)
    while True:
        if datetime.now() >= to_compare:
            TEXT_TO_SEND={ 
                    "id": DEVICE_ID, 
                    "d": "1.0.0,100,1,1"                   
                    } 
            TEXT_TO_SEND = json.dumps(TEXT_TO_SEND, indent = 4) 
            #TEXT_TO_SEND = f'id: {DEVICE_ID},"d":"1.0.0,100,1,1"'
            #TEXT_TO_SEND=json.loads(TEXT_TO_SEND)
            send_to_iot_hub(CLIENT,TEXT_TO_SEND)
            to_compare = datetime.now() + timedelta(minutes=minutes_freq)
            try : ret, img = cam.read()
            except : pass
            if not ret:
                try:                
                    data = json.loads(open("/home/video_source.json", 'r').read())
                    video_source=data['source']
                    """startPoint_X = int(data['startPoint_X'])
                    startPoint_Y = int(data['startPoint_Y'])
                    endPoint_X = int(data['endPoint_X'])
                    endPoint_Y = int(data['endPoint_Y'])"""
                    in_X1= int(data['in_X1'])
                    in_Y1= int(data['in_Y1'])
                    in_X2= int(data['in_X2'])
                    in_Y2= int(data['in_Y2'])
                    out_X1 =int(data['out_X1'])
                    out_X2=int(data['out_X2'])
                    out_Y1= int(data['out_Y1'])
                    out_Y2= int(data['out_Y2'])
                    #in1, in2, counter = (in_X1, in_Y1), (in_X2, in_X2), 0 #(10,784), (1750,784), 0
                    out1,out2=(out_X1, out_Y1), (out_X2, out_Y2)
                    in1= (in_X1, in_Y1)
                    in2=(in_X2, in_Y2)
                    roi_1=str(in1)+str(in2)
                    roi_2=str(out1)+str(out2)
                    skip_frames= int(data['skip_frames'])
                    minutes_freq = int(data['minutes_freq'])
                    freq = int(data['freq_iot_hub'])
                    cip= data['cip']
                    day_Confidence=float(data['day_Confidence'])
                    night_Confidence=float(data['night_Confidence'])
                    start_Daytime=int(data['start_Daytime'])
                    end_Daytime=int(data['end_Daytime'])
                    cam= cv2.VideoCapture(video_source)
                    frame_width = int(cam.get(3))
                    frame_height = int(cam.get(4))
                    
                    size = (frame_width, frame_height)
                    frameIndex = 0

                    (W, H) = (None, None)
                    (ret, img) = cam.read()
                    #print(camera_Disconnect)
                    camera_Disconnect=0
                    #roi=str(start_point) + str(end_point)
                    #print(camera_Disconnect)
                    time_Now=str(datetime.now().strftime("%m/%d/%Y %H:%M:%S"))
                    TEXT_TO_SEND={ 
                            "id": DEVICE_ID, 
                            "d": "1.0.0,100,1,1", 
                            "t": time_Now,
                            "status":"Camera_offline",
                            "cip":video_source,
                            "roi_in":roi_in,
                            "roi_out":roi_out                   
                            } 
                    TEXT_TO_SEND = json.dumps(TEXT_TO_SEND, indent = 4) #= json.dumps(dictionary, indent = 4) 
                    #TEXT_TO_SEND = '{'+'"id":"'+str(DEVICE_ID)+'","d":"1.0.0,100,1,1"'+',"status": "camera_offline","t":"'+str(time_Now)+'"}'
                    
                    send_to_iot_hub(CLIENT,TEXT_TO_SEND)
                    continue
                except:
                    camera_Disconnect+=1
                    #print(camera_Disconnect)
                    try:
                        cam = cv2.VideoCapture(video_source)
                        data = json.loads(open("/home/video_source.json", 'r').read())
                        video_source=data['source']
                        """startPoint_X = int(data['startPoint_X'])
                        startPoint_Y = int(data['startPoint_Y'])
                        endPoint_X = int(data['endPoint_X'])
                        endPoint_Y = int(data['endPoint_Y'])"""
                        in_X1= int(data['in_X1'])
                        in_Y1= int(data['in_Y1'])
                        in_X2= int(data['in_X2'])
                        in_Y2= int(data['in_Y2'])
                        out_X1 =int(data['out_X1'])
                        out_X2=int(data['out_X2'])
                        out_Y1= int(data['out_Y1'])
                        out_Y2= int(data['out_Y2'])
                        #in1, in2, counter = (in_X1, in_Y1), (in_X2, in_X2), 0 #(10,784), (1750,784), 0
                        out1,out2=(out_X1, out_Y1), (out_X2, out_Y2)
                        in1= (in_X1, in_Y1)
                        in2=(in_X2, in_Y2)
                        roi_1=str(in1)+str(in2)
                        roi_2=str(out1)+str(out2)
                        skip_frames= int(data['skip_frames'])
                        minutes_freq = int(data['minutes_freq'])
                        freq = int(data['freq_iot_hub'])
                        day_Confidence=float(data['day_Confidence'])
                        night_Confidence=float(data['night_Confidence'])
                        start_Daytime=int(data['start_Daytime'])
                        end_Daytime=int(data['end_Daytime'])
                        cip= data['cip']
                        cam =cv2.VideoCapture(video_source)# video_source#=#
                        frame_width = int(cam.get(3))
                        frame_height = int(cam.get(4))
                        
                        size = (frame_width, frame_height)
                        frameIndex = 0

                        #(W, H) = (None, None)
                        (ret,img) = cam.read()
                    except :pass
                    
                    camera_Disconnect=0
                    #print(camera_Disconnect)
                    time_Now=str(datetime.now().strftime("%m/%d/%Y %H:%M:%S"))
                    #TEXT_TO_SEND = '{'+'"id":"'+str(DEVICE_ID)+'","d":"1.0.0,100,1,1"'+',"status": "camera_offline","t":"'+str(time_Now)+'"}'
                    TEXT_TO_SEND={ 
                            "id": DEVICE_ID, 
                            "d": "1.0.0,100,1,1", 
                            "t": time_Now,
                            "status":"Camera_offline",
                            "cip":video_source,
                            "roi_in":roi_in,
                            "roi_out":roi_out  
                            } 
                    TEXT_TO_SEND = json.dumps(TEXT_TO_SEND, indent = 4) 
                    #send_to_iot_hub(CLIENT,TEXT_TO_SEND)
                    #camera_Disconnect=0
                    continue
        
        now=datetime.now()
        now=now.strftime("%H:%M:%S")
        if int(now[1])==0 and int(now[0])==0:
            if os.path.exists('/home/logs.txt'):
                os.remove('/home/logs.txt')
            time.sleep(60)
            subprocess.run(["sudo","reboot"])
        if int(now[1])>start_Daytime and int(now[1]) <end_Daytime:#3,15
            confidence=day_Confidence
            #print("daytime"+str(confidence))
        else: confidence=night_Confidence

        try : ret, img = cam.read()
        except : pass
        if not ret:
            try:                
                data = json.loads(open("/home/video_source.json", 'r').read())
                video_source=data['source']
                startPoint_X = int(data['startPoint_X'])
                startPoint_Y = int(data['startPoint_Y'])
                endPoint_X = int(data['endPoint_X'])
                endPoint_Y = int(data['endPoint_Y'])
                skip_frames= int(data['skip_frames'])
                in_X1= int(data['in_X1'])
                in_Y1= int(data['in_Y1'])
                in_X2= int(data['in_X2'])
                in_Y2= int(data['in_Y2'])
                out_X1 =int(data['out_X1'])
                out_X2=int(data['out_X2'])
                out_Y1= int(data['out_Y1'])
                out_Y2= int(data['out_Y2'])
                #in1, in2, counter = (in_X1, in_Y1), (in_X2, in_X2), 0 #(10,784), (1750,784), 0
                out1,out2=(out_X1, out_Y1), (out_X2, out_Y2)
                in1= (in_X1, in_Y1)
                in2=(in_X2, in_Y2)
                minutes_freq = int(data['minutes_freq'])
                freq = int(data['freq_iot_hub'])
                cip= data['cip']
                day_Confidence=float(data['day_Confidence'])
                night_Confidence=float(data['night_Confidence'])
                start_Daytime=int(data['start_Daytime'])
                end_Daytime=int(data['end_Daytime'])
                cam= cv2.VideoCapture(video_source)
                frame_width = int(cam.get(3))
                frame_height = int(cam.get(4))
                
                size = (frame_width, frame_height)
                frameIndex = 0

                (W, H) = (None, None)
                (ret, img) = cam.read()
                #print(camera_Disconnect)
                camera_Disconnect+=1
                if camera_Disconnect==10000:
                    camera_Disconnect=0

                    #print(camera_Disconnect)
                    time_Now=str(datetime.now().strftime("%m/%d/%Y %H:%M:%S"))
                    TEXT_TO_SEND={ 
                            "id": DEVICE_ID, 
                            "d": "1.0.0,100,1,1", 
                            "t": time_Now,
                            "status":"Camera_offline",
                            "roi_in":roi_in,
                            "roi_out":roi_out                    
                            } 
                    TEXT_TO_SEND = json.dumps(TEXT_TO_SEND, indent = 4) #= json.dumps(dictionary, indent = 4) 
                    #TEXT_TO_SEND = '{'+'"id":"'+str(DEVICE_ID)+'","d":"1.0.0,100,1,1"'+',"status": "camera_offline","t":"'+str(time_Now)+'"}'
                    
                    #send_to_iot_hub(CLIENT,TEXT_TO_SEND)
                continue
            except:
                camera_Disconnect+=1
                #print(camera_Disconnect)
                try:
                    cam = cv2.VideoCapture(video_source)
                    data = json.loads(open("/home/video_source.json", 'r').read())
                    video_source=data['source']
                    """startPoint_X = int(data['startPoint_X'])
                    startPoint_Y = int(data['startPoint_Y'])
                    endPoint_X = int(data['endPoint_X'])
                    endPoint_Y = int(data['endPoint_Y'])"""
                    in_X1= int(data['in_X1'])
                    in_Y1= int(data['in_Y1'])
                    in_X2= int(data['in_X2'])
                    in_Y2= int(data['in_Y2'])
                    out_X1 =int(data['out_X1'])
                    out_X2=int(data['out_X2'])
                    out_Y1= int(data['out_Y1'])
                    out_Y2= int(data['out_Y2'])
                    #in1, in2, counter = (in_X1, in_Y1), (in_X2, in_X2), 0 #(10,784), (1750,784), 0
                    out1,out2=(out_X1, out_Y1), (out_X2, out_Y2)
                    in1= (in_X1, in_Y1)
                    in2=(in_X2, in_Y2)
                    skip_frames= int(data['skip_frames'])
                    minutes_freq = int(data['minutes_freq'])
                    freq = int(data['freq_iot_hub'])
                    day_Confidence=float(data['day_Confidence'])
                    night_Confidence=float(data['night_Confidence'])
                    start_Daytime=int(data['start_Daytime'])
                    end_Daytime=int(data['end_Daytime'])
                    cip= data['cip']
                    cam =cv2.VideoCapture(video_source)# video_source#=#
                    frame_width = int(cam.get(3))
                    frame_height = int(cam.get(4))
                    
                    size = (frame_width, frame_height)
                    frameIndex = 0

                    #(W, H) = (None, None)
                    (ret,img) = cam.read()
                except :pass
                if camera_Disconnect==10000:
                    camera_Disconnect=0
                    #print(camera_Disconnect)
                    time_Now=str(datetime.now().strftime("%m/%d/%Y %H:%M:%S"))
                    #TEXT_TO_SEND = '{'+'"id":"'+str(DEVICE_ID)+'","d":"1.0.0,100,1,1"'+',"status": "camera_offline","t":"'+str(time_Now)+'"}'
                    TEXT_TO_SEND={ 
                            "id": DEVICE_ID, 
                            "d": "1.0.0,100,1,1", 
                            "t": time_Now,
                            "status":"Camera_offline",
                            "roi_in":roi_in,
                            "roi_out":roi_out                    
                            } 
                    TEXT_TO_SEND = json.dumps(TEXT_TO_SEND, indent = 4) 
                    #send_to_iot_hub(CLIENT,TEXT_TO_SEND)
                    camera_Disconnect=0
                continue
        camera_Disconnect=0 
        #if frameIndex % (skip_frames) == 0:

        """if img is None:
            (H, W) = img.shape[:2]
            continue"""
        boxes, confs, clss = trt_yolo.detect(img,confidence)
        boxes_=boxes.tolist()
        idxs = cv2.dnn.NMSBoxes(boxes_,confs, confidence,confidence)
        dets=[]
        if len(idxs) > 0:
            for box,conf,clsss in zip(boxes,confs,clss):
                #print(clsss)
                #if clsss in [0,1,2,3,5,7]:
                dets.append([box[0], box[1], box[2],box[3], conf])
                #cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 2)

        try:
            np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
            dets = np.asarray(dets)
            tracks = tracker.update(dets)
            #print(tracks)
        except: pass
        boxes = []
        indexIDs = []
        c = []
        previous = memory.copy()
        #print("previous")
        #print(previous)
        memory = {}
        #cv2.imshow('frame',frame)
        for track in tracks:
            boxes.append([track[0], track[1], track[2], track[3]])
            indexIDs.append(int(track[4]))
            memory[indexIDs[-1]] = boxes[-1]
        #print(boxes)
        if len(boxes) >=0:
            i = int(0)
            for box in boxes:
                #print("box")
                #print(box)
                # extract the bounding box coordinates
                (x, y) = (int(box[0]), int(box[1]))
                (w, h) = (int(box[2]), int(box[3]))

                if indexIDs[i] in previous:
                    previous_box = previous[indexIDs[i]]
                    (x2, y2) = (int(previous_box[0]), int(previous_box[1]))
                    (w2, h2) = (int(previous_box[2]), int(previous_box[3]))
                    p0 = (int(x + (w-x)/2), int(y + (h-y)/2))
                    p1 = (int(x2 + (w2-x2)/2), int(y2 + (h2-y2)/2))
                    
                    if intersect(p0, p1,in1, in2):
                        time_Now=str(datetime.now().strftime("%m/%d/%Y %H:%M:%S"))
                        #with open('sample.txt', 'r') as f:
                            #temp_counter=f.read()
                        #counter=int(temp_counter)

                        counter1 += 1
                        if TEXT_TO_SEND == "":
                            #roi=str(start_point) + str(end_point)
                            TEXT_TO_SEND={ 
                            "id": device_id,
                            "d": "1.0.0,100,1,1",
                            "t": time_Now,
                            "n":1,
                            "c":1,
                            "cip":video_source,
                            "roi_in":roi_in,
                            "roi_out":roi_out,
                            "s":"in"                                     
                            } 
                            TEXT_TO_SEND = json.dumps(TEXT_TO_SEND, indent = 4) 
                            #TEXT_TO_SEND = '{'+'"id":"'+str(DEVICE_ID)+'","d":"1.0.0,100,1,1","t":"'+str(time_Now)+'","n":1,"c":1,"cip":"'+str(cip)+'","roi":"'+str(roi)+'"}'
                            #TEXT_TO_SEND=json.loads(TEXT_TO_SEND)
                            send_to_iot_hub(CLIENT,TEXT_TO_SEND)
                            #TEXT_TO_SEND = '{'+'"id":"'+str(DEVICE_ID)+'","d":"1.0.0,100,1,1","t":"'+str(time_Now)+'","n":1,"c":"'+count+',"cip":"'+str(cip)+'","roi":"'+str(roi)+'"}'
                            #TEXT_TO_SEND = '{'+'"id":"'+str(DEVICE_ID)+'","d":"1.0.0,100,1,1","t":"'+str(time_Now)+'","n":"1","c":"'+str(count)+',"cip":"'+str(cip)+'","roi":"'+str(roi)+'"}'
                            
                            TEXT_TO_SEND == ""
                        else:
                            count=1                                 
                            #roi=str(start_point) + str(end_point)
                            #TEXT_TO_SEND = '{'+'"id":"'+str(DEVICE_ID)+'","d":"1.0.0,100,1,1","t":"'+str(time_Now)+'","n":1,"c":1,"cip":"'+str(cip)+'","roi":"'+str(roi)+'"}'
                            #TEXT_TO_SEND=json.loads(TEXT_TO_SEND)
                            TEXT_TO_SEND={ 
                            "id": device_id,
                            "d": "1.0.0,100,1,1",
                            "t": time_Now,
                            "n":1,
                            "c":1,
                            "cip":video_source,
                            "roi_in":roi_in,
                            "roi_out":roi_out,
                            "s":"in"               
                            } 
                            TEXT_TO_SEND = json.dumps(TEXT_TO_SEND, indent = 4) 
                            send_to_iot_hub(CLIENT,TEXT_TO_SEND)
                            #TEXT_TO_SEND = '{'+'"id":"'+str(DEVICE_ID)+'","d":"1.0.0,100,1,1","t":"'+str(time_Now)+'","n":"1","c":'+count+',"cip":"'+str(cip)+'","roi":"'+str(roi)+'"}'
                            #TEXT_TO_SEND = '{'+'"id":"'+str(DEVICE_ID)+'","d":"1.0.0,100,1,1","t":"'+str(time_Now)+'","n":"1","c":"'+str(count)+',"cip":"'+str(cip)+'","roi":"'+str(roi)+'"}'
                            TEXT_TO_SEND == ""
                    if intersect(p0, p1,out1, out2):
                        time_Now=str(datetime.now().strftime("%m/%d/%Y %H:%M:%S"))
                        #with open('sample.txt', 'r') as f:
                            #temp_counter=f.read()
                        #counter=int(temp_counter)

                        counter2 += 1
                        if TEXT_TO_SEND == "":
                            #roi=str(748,431) + str(1266,479)
                            TEXT_TO_SEND={ 
                            "id": device_id,
                            "d": "1.0.0,100,1,1",
                            "t": time_Now,
                            "n":1,
                            "c":1,
                            "cip":video_source,
                            "roi_out":roi_out,
                            "s":"out"  
                                   
                            } 
                            TEXT_TO_SEND = json.dumps(TEXT_TO_SEND, indent = 4) 
                            #TEXT_TO_SEND = '{'+'"id":"'+str(DEVICE_ID)+'","d":"1.0.0,100,1,1","t":"'+str(time_Now)+'","n":1,"c":1,"cip":"'+str(cip)+'","roi":"'+str(roi)+'"}'
                            #TEXT_TO_SEND=json.loads(TEXT_TO_SEND)
                            send_to_iot_hub(CLIENT,TEXT_TO_SEND)
                            #TEXT_TO_SEND = '{'+'"id":"'+str(DEVICE_ID)+'","d":"1.0.0,100,1,1","t":"'+str(time_Now)+'","n":1,"c":"'+count+',"cip":"'+str(cip)+'","roi":"'+str(roi)+'"}'
                            #TEXT_TO_SEND = '{'+'"id":"'+str(DEVICE_ID)+'","d":"1.0.0,100,1,1","t":"'+str(time_Now)+'","n":"1","c":"'+str(count)+',"cip":"'+str(cip)+'","roi":"'+str(roi)+'"}'
                            
                            TEXT_TO_SEND == ""
                        else:
                            count=1                                 
                            #roi=str(start_point) + str(end_point)
                            #TEXT_TO_SEND = '{'+'"id":"'+str(DEVICE_ID)+'","d":"1.0.0,100,1,1","t":"'+str(time_Now)+'","n":1,"c":1,"cip":"'+str(cip)+'","roi":"'+str(roi)+'"}'
                            #TEXT_TO_SEND=json.loads(TEXT_TO_SEND)
                            TEXT_TO_SEND={ 
                            "id": device_id,
                            "d": "1.0.0,100,1,1",
                            "t": time_Now,
                            "n":1,
                            "c":1,
                            "cip":video_source,
                            "roi_out":roi_out,
                            "s":"in"                
                            } 
                            TEXT_TO_SEND = json.dumps(TEXT_TO_SEND, indent = 4) 
                            send_to_iot_hub(CLIENT,TEXT_TO_SEND)
                            #TEXT_TO_SEND = '{'+'"id":"'+str(DEVICE_ID)+'","d":"1.0.0,100,1,1","t":"'+str(time_Now)+'","n":"1","c":'+count+',"cip":"'+str(cip)+'","roi":"'+str(roi)+'"}'
                            #TEXT_TO_SEND = '{'+'"id":"'+str(DEVICE_ID)+'","d":"1.0.0,100,1,1","t":"'+str(time_Now)+'","n":"1","c":"'+str(count)+',"cip":"'+str(cip)+'","roi":"'+str(roi)+'"}'
                            TEXT_TO_SEND == ""
                    
                        
                    i += 1
            print(counter1)
            print(counter2)
        if frameIndex % 1000 == 1:
            temp_data = json.loads(open('/home/video_source.json', 'r').read())
            if temp_data!=data:
                data = temp_data
                source = data['source']
                """startPoint_X = int(data['startPoint_X'])
                startPoint_Y = int(data['startPoint_Y'])
                endPoint_X = int(data['endPoint_X'])
                endPoint_Y = int(data['endPoint_Y'])"""
                in_X1= int(data['in_X1'])
                in_Y1= int(data['in_Y1'])
                in_X2= int(data['in_X2'])
                in_Y2= int(data['in_Y2'])
                out_X1 =int(data['out_X1'])
                out_X2=int(data['out_X2'])
                out_Y1= int(data['out_Y1'])
                out_Y2= int(data['out_Y2'])
                out1,out2=(out_X1, out_Y1), (out_X2, out_Y2)
                in1= (in_X1, in_Y1)
                in2=(in_X2, in_Y2)
                cip= data['cip']
                skip_frames= int(data['skip_frames'])
                minutes_freq = int(data['minutes_freq'])
                freq = int(data['freq_iot_hub'])
                day_Confidence=float(data['day_Confidence'])
                night_Confidence=float(data['night_Confidence'])
                start_Daytime=int(data['start_Daytime'])
                end_Daytime=int(data['end_Daytime'])
                start_point = (startPoint_X, startPoint_Y)
                end_point = (endPoint_X, endPoint_Y)
                CAMERA_SOURCE = source
                cam = cv2.VideoCapture(source)
                frame_width = int(cam.get(3))
                frame_height = int(cam.get(4))
                
                size = (frame_width, frame_height)
                frameIndex = 0
                (ret, img) = cam.read()
                roi_in=str(in1) + str(in2)
                roi_out=str(out1) + str(out2)
                TEXT_TO_SEND={ 
                            "id": DEVICE_ID, 
                            "d": "1.0.0,100,1,1", 
                            "t": time_Now,
                            "status":"new_config_received",
                            "cip":video_source,
                            "roi_in":roi_in,
                            "roi_out":roi_out                  
                } 
                TEXT_TO_SEND = json.dumps(TEXT_TO_SEND, indent = 4) 
                #TEXT_TO_SEND = '{'+'"id":"'+str(DEVICE_ID)+'","cip":"'+str(cip)+'","status":"new_config_received","t":"'+str(time_Now)+'"}'
                
                send_to_iot_hub(CLIENT,TEXT_TO_SEND)
                try:
                    if img is None:
                        #roi=str(start_point) + str(end_point)
                        roi_in=str(in1) + str(in2)
                        roi_out=str(out1) + str(out2)
                        time_Now=str(datetime.now().strftime("%m/%d/%Y %H:%M:%S"))
                        TEXT_TO_SEND={ 
                                "id": DEVICE_ID, 
                                "d": "1.0.0,100,1,1", 
                                "t": time_Now,
                                "status":"invalid_camera",
                                "cip":video_source,
                                "roi_in":roi_in,
                                "roi_out":roi_out                      
                        } 
                        TEXT_TO_SEND = json.dumps(TEXT_TO_SEND, indent = 4) 
                        #TEXT_TO_SEND = '{'+'"id":"'+str(DEVICE_ID)+'","cip":"'+str(cip)+'","status":"invalid_camera","t":"'+str(time_Now)+'"}'
                        send_to_iot_hub(CLIENT,TEXT_TO_SEND)

                    else: 
                        #roi=str(start_point) + str(end_point)
                        roi_in=str(in1) + str(in2)
                        roi_out=str(out1) + str(out2)
                        time_Now=str(datetime.now().strftime("%m/%d/%Y %H:%M:%S"))
                        TEXT_TO_SEND={ 
                                "id": DEVICE_ID, 
                                "d": "1.0.0,100,1,1", 
                                "t": time_Now,
                                "status":"valid_camera",
                                "cip":video_source,
                                "roi_in":roi_in,
                                "roi_out":roi_out,

                        } 
                        TEXT_TO_SEND = json.dumps(TEXT_TO_SEND, indent = 4) 
                        #TEXT_TO_SEND = '{'+'"id":"'+str(DEVICE_ID)+'","cip":"'+str(cip)+'","status":"valid_camera","t":"'+str(time_Now)+'"}'
                        send_to_iot_hub(CLIENT,TEXT_TO_SEND)
                except: pass
        if frameIndex % 1001 == 0:
            try:
                subprocess.run(["sudo","ifmetric","wwan0","48"])
            except:pass
        """if frameIndex ==4000:
            out.release()
            cam.release()
            break"""
        if frameIndex % (freq*30*60 + 1)==0 or "status" in TEXT_TO_SEND:
            if TEXT_TO_SEND != "":
                TEXT_TO_SEND = ""
        if frameIndex == 0:
            #roi=str(start_point) + str(end_point)
            #now = datetime.now

            time_Now=str(datetime.now().strftime("%m/%d/%Y %H:%M:%S"))
            TEXT_TO_SEND={ 
                            "id": DEVICE_ID, 
                            "d": "1.0.0,100,1,1", 
                            "t": time_Now,
                            "status":"connecting-camera",
                            "cip": video_source,
                            "roi_in": roi_in,
                            "roi_out": roi_out
                                          
                    } 
            TEXT_TO_SEND = json.dumps(TEXT_TO_SEND, indent = 4) 
            #TEXT_TO_SEND = '{'+'"id":"'+str(DEVICE_ID)+'",cip":"'+str(cip)+'","status": "connecting-camera","t":"'+str(time_Now)+'"}'
            send_to_iot_hub(CLIENT,TEXT_TO_SEND)
            

        frameIndex = frameIndex+1
        """print (frameIndex)
        color = (0, 255, 255)
        cv2.line(img, in1, in2, color, 2)
        cv2.line(img, out1, out2, color, 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
  
        # org
        org = (50, 50)
        
        # fontScale
        fontScale = 2
        
        # Blue color in BGR
        color = (255, 0, 0)
        cv2.putText(img, "in : "+str(counter1), org, font, 
                   fontScale, color,2, cv2.LINE_AA)
        cv2.putText(img, "out : "+str(counter2), (1000,50), font, 
                   fontScale, color,2, cv2.LINE_AA)
        cv2.imwrite('frame.jpg',img)
        out.write(img)"""



   

def main():
    with open('/home/mac.txt','r') as file:
        device_id = file.read()
        device_id=device_id.rstrip()
        device_id=device_id.lstrip()
    with open('/home/cs.txt','r') as file2:
        cs = file2.read()
        cs=cs.rstrip()
        cs=cs.lstrip()
    CONNECTION_STRING = cs 
    CLIENT = IoTHubDeviceClient.create_from_connection_string(CONNECTION_STRING)
    TEXT_TO_SEND = ""


    camera_Disconnect = 0
    frameIndex = 0
    DEVICE_ID = device_id #os.getenv("id")
    data = json.loads(open("/home/video_source.json", 'r').read())
    video_source = data['source']
    startPoint_X = int(data['startPoint_X'])
    startPoint_Y = int(data['startPoint_Y'])
    endPoint_X = int(data['endPoint_X'])
    endPoint_Y = int(data['endPoint_Y'])
    start_point = (startPoint_X, startPoint_Y)
    end_point = (endPoint_X, endPoint_Y)
    skip_frames= int(data['skip_frames'])
    minutes_freq = int(data['minutes_freq'])
    freq = int(data['freq_iot_hub'])
    #confidence=float(data['confidence'])
    day_Confidence=float(data['day_Confidence'])
    night_Confidence=float(data['night_Confidence'])
    start_Daytime=int(data['start_Daytime'])
    end_Daytime=int(data['end_Daytime'])
    cip= data['cip']
    start_point, end_point, counter = (startPoint_X, startPoint_Y), (endPoint_X, endPoint_Y), 0 #(10,784), (1750,784), 0
    camera_Disconnect=0
    now=datetime.now()
    now=now.strftime("%H:%M:%S")
    in_X1= int(data['in_X1'])
    in_Y1= int(data['in_Y1'])
    in_X2= int(data['in_X2'])
    in_Y2= int(data['in_Y2'])
    out_X1 =int(data['out_X1'])
    out_X2=int(data['out_X2'])
    out_Y1= int(data['out_Y1'])
    out_Y2= int(data['out_Y2'])
    #in1, in2, counter = (in_X1, in_Y1), (in_X2, in_X2), 0 #(10,784), (1750,784), 0
    out1,out2=(out_X1, out_Y1), (out_X2, out_Y2)
    in1= (in_X1, in_Y1)
    in2=(in_X2, in_Y2)
    roi_in=str(in1)+str(in2)
    roi_out=str(out1)+str(out2)

    cls_dict = get_cls_dict(args.category_num)
    vis = BBoxVisualization(cls_dict)
    trt_yolo = TrtYOLO(args.model, args.category_num, args.letter_box)

    
    if int(now[1])>start_Daytime and int(now[1]) <end_Daytime:#3,15
        confidence=day_Confidence
        #print("daytime"+str(confidence))
    else: confidence=night_Confidence
    loop_and_detect(cam, trt_yolo, confidence, vis=vis)
    
    """cam.release()
    out.release()
    cv2.destroyAllWindows()"""


if __name__ == '__main__':
    data = json.loads(open("/home/video_source.json", 'r').read())
    video_source = data['source']
    args = parse_args()
    cam=cv2.VideoCapture(video_source)
    """w=int(cam.get(3))
    h=int(cam.get(4))
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc('X','V','I','D')
    out = cv2.VideoWriter('output.avi',fourcc, 20.0, (w,h))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')"""
    #out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))
    #print("out")
    main()
