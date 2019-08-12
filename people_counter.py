# Modified from good work done by Adrian Rosebrock https://www.pyimagesearch.com/2018/08/13/opencv-people-counter/

# import the necessary packages
from pyimagesearch.centroidtracker import CentroidTracker
from pyimagesearch.trackableobject import TrackableObject
from imutils.video import WebcamVideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import base64
from threading import Thread
import sys
import json
import signal

# Import AWS IoT SDK package
from AWSIoTPythonSDK.MQTTLib import AWSIoTMQTTClient

# open the config.json file and load the config settings into the var
with open('config.json') as config_file:
    configs = json.load(config_file)

def keyboardInterruptHandler(signal, frame):
    print("KeyboardInterrupt (ID: {}) has been caught. Cleaning up...".format(signal))
    msg = {}
    msg["message"] = "streamState"
    msg["state"] = "stopped"
    jsonMsg = json.dumps(msg)
    myMQTTClient.publishAsync(
        "human/" + configs['iotDeviceName'] + "/status", jsonMsg, 0, ackCallback=iotResponse)

    # check to see if we need to release the video writer pointer
    if writer is not None:
        writer.release()

    print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

    # if we are using a video file, stop
    if configs['useVideoStream'] == 0:
        fvs.release()
    # otherwise, release the video file pointer
    else:
        fvs.stop()

    # close any open windows
    cv2.destroyAllWindows()
    msg = {}
    msg['message'] = "Application"
    msg['state'] = "stopped"
    jsonMsg = json.dumps(msg)
    myMQTTClient.publishAsync(
        "human/" + configs['iotDeviceName'] + "/status", jsonMsg, 0, ackCallback=iotResponse)
    myMQTTClient.disconnect()
    cv2.destroyAllWindows()
    sys.exit()
    exit(0)

signal.signal(signal.SIGINT, keyboardInterruptHandler)

# Suback callback
def subackCallback(mid, data):
    print("subAck")

def onMessage(message):
    print("onMessage")

# Puback callback
def iotResponse(mid):
    print("Puback")

# Set Up AWS iot
myMQTTClient = AWSIoTMQTTClient(configs['iotDeviceName'])
myMQTTClient.configureEndpoint(configs['iotEndpoint'], configs['iotPort'])
myMQTTClient.configureCredentials(
    configs['iotCACert'], configs['iotPrivCert'], configs['iotCert'])
myMQTTClient.configureAutoReconnectBackoffTime(
    configs['baseReconnectQuietTimeSecond'], configs['maxReconnectQuietTimeSecond'], configs['stableConnectionTimeSecond'])
myMQTTClient.configureOfflinePublishQueueing(-1)
myMQTTClient.configureDrainingFrequency(2)
myMQTTClient.configureConnectDisconnectTimeout(configs['iotTimeout'])
myMQTTClient.configureMQTTOperationTimeout(configs['iotOpTimeout'])
myMQTTClient.onMessage = onMessage

# Now lets do connection
myMQTTClient.connect()

myMQTTClient.subscribeAsync("human/" + configs['iotDeviceName'] + "/getConfig", 0, ackCallback=subackCallback)
time.sleep(2)

# Push the config to AWS IoT so its available online to allow checking of config settings
# IoT Platform can then alert to any changes without the need to use a shadow and can push messages
msg = {}
msg['message'] = "Config"
msg['config'] = configs
jsonMsg = json.dumps(msg)
myMQTTClient.publishAsync(
    "human/" + configs['iotDeviceName'] + "/config", jsonMsg, 0, ackCallback=iotResponse)

msg = {}
msg['message'] = "Application"
msg['state'] = "started"
jsonMsg = json.dumps(msg)
myMQTTClient.publishAsync(
    "human/" + configs['iotDeviceName'] + "/status", jsonMsg, 0, ackCallback=iotResponse)

# initialize the list of class labels MobileNet SSD was trained to
# detect
# TODO: need to change for YOLO if using.
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

# load our serialized model from disk
print("[INFO] loading model...")

# Switch based on API/network.  Not fully implemented.  Only Caffee in use in this version

if configs['modelType'] == "readNetFromCaffe":
    net = cv2.dnn.readNetFromCaffe(
        configs['modelPath'] + configs['modelConfig'], configs['modelPath'] + configs['modelFile'])
elif configs['modelType'] == "readNetFromDarknet":
    net = cv2.dnn.readNetFromDarknet(
        configs['modelPath'] + configs['modelConfig'], configs['modelPath'] + configs['modelFile'])
elif configs['modelType'] == "readNetFromTensorflow":
    net = cv2.dnn.readNetFromTensorflow(
        configs['modelPath'] + configs['modelConfig'], configs['modelPath'] + configs['modelFile'])
elif configs['modelType'] == "readNetFromTorch":
    net = cv2.dnn.readNetFromTorch(
        configs['modelPath'] + configs['modelConfig'], configs['modelPath'] + configs['modelFile'])
else:
    print("[CRITICAL] Couldnt work out the model...could be wrong call for wrong model file or just unsupported framework")

# if a video path was not supplied, grab a reference to the usb camera
if configs['useVideoStream'] == 0:
    print("[INFO] opening file...")
    fvs = cv2.VideoCapture(configs['videoFilePath'])
    time.sleep(2.0)
# otherwise, grab a reference to the video file
else:
    print("[INFO] opening stream...")
    fvs = cv2.VideoCapture(configs['videoStream'])

msg = {}
msg['message'] = "streamState"
msg['streamState'] = "started"
jsonMsg = json.dumps(msg)
myMQTTClient.publishAsync(
    "human/" + configs['iotDeviceName'] + "/status", jsonMsg, 0, ackCallback=iotResponse)

# initialize the video writer
writer = None

# initialize the frame dimensions (we'll set them as soon as we read
# the first frame from the video)
W = None
H = None

# instantiate the centroid tracker, then initialize a list to store
# each of our dlib correlation trackers, followed by a dictionary to
# map each unique object ID to a TrackableObject
ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
trackers = []
trackableObjects = {}

# initialize the total number of frames processed thus far, along
# with the total number of objects that have moved either up or down
totalFrames = 0
totalDown = 0
totalUp = 0
status = 0

# vars used for storing the up and down counts that are pushed every 30 seconds to AWS IoT
iotUp = 0
iotDown = 0
iotFrameCnt = 0

totalUpPrevious = 0
totalDownPrevious = 0

# start the frames per second throughput estimator
fps = FPS().start()

try:
    # loop over frames from the video stream
    while True:
        # increment for each frame to keep tally of the number of frames and reset later when 180 hit (about 30 seconds)
        iotFrameCnt += 1

        # grab the next frame and handle if we are reading from either
        # VideoCapture or VideoStream
        frame = fvs.read()
        # only handles video stream of file but not webcam as not needed
        frame = frame[1]
        #pushFrame = frame
        #pushFrame = imutils.resize(frame,width=150)

        if frame is None:
            break

        frame = imutils.resize(frame,width=500)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # frame = np.dstack([frame, frame, frame])

        # show the frame and update the FPS counter
        if configs['showScreen'] == 1:
            cv2.imshow("Frame", frame)
            cv2.waitKey(1)

        # if we are viewing a video and we did not grab a frame then we
        # have reached the end of the video
        if frame is None:
            msg = {}
            msg["message"] = "streamState"
            msg["state"] = "stopped"
            jsonMsg = json.dumps(msg)
            myMQTTClient.publishAsync(
                "human/" + configs['iotDeviceName'] + "/status", jsonMsg, 0, ackCallback=iotResponse)
            break

        # if the frame dimensions are empty, set them
        if W is None or H is None:
            (H, W) = frame.shape[:2]

        # if we are supposed to be writing a video to disk, initialize
        # the writer
        if configs['captureVideo'] == 1 and writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(
                configs['captureVideoPath'], fourcc, 30, (W, H), True)

        # initialize the current status along with our list of bounding
        # box rectangles returned by either (1) our object detector or
        # (2) the correlation trackers
        # status = "Waiting"
        rects = []

        # check to see if we should run a more computationally expensive
        # object detection method to aid our tracker
        if totalFrames % configs['skipFrames'] == 0:
            # set the status and initialize our new set of object trackers
            status = "Detecting"
            trackers = []

            # convert the frame to a blob and pass the blob through the
            # network and obtain the detections. Darknet is a bit different. Others to be added later
            if configs['modelType'] == "readNetFromDarknet":
                blob = cv2.dnn.blobFromImage(frame, 1.0/255.0, (416, 416), False, crop=False)
                net.setInput(blob)
                detections = net.forward()
                probability_index=5
                # loop over the detections
                for i in range(detections.shape[0]):
                    # extract the confidence (i.e., probability) associated
                    # with the prediction
                    confidence=np.amax(prob_arr)
                    # confidence = detections[0, 0, i, 2]

                    # filter out weak detections by requiring a minimum
                    # confidence
                    if confidence > configs['confidence']:
                        # extract the index of the class label from the
                        # detections list
                        idx = int(detections[0, 0, i, 1])

                        # if the class label is not a person, ignore it
                        if CLASSES[idx] != "person":
                            continue

                        # compute the (x, y)-coordinates of the bounding box
                        # for the object
                        box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                        (startX, startY, endX, endY) = box.astype("int")

                        # construct a dlib rectangle object from the bounding
                        # box coordinates and then start the dlib correlation
                        # tracker
                        tracker = dlib.correlation_tracker()
                        rect = dlib.rectangle(startX, startY, endX, endY)
                        tracker.start_track(rgb, rect)

                        # add the tracker to our list of trackers so we can
                        # utilize it during skip frames
                        trackers.append(tracker)
            else:
                blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
                net.setInput(blob)
                detections = net.forward()
                # loop over the detections
                for i in np.arange(0, detections.shape[2]):
                    # extract the confidence (i.e., probability) associated
                    # with the prediction
                    confidence = detections[0, 0, i, 2]

                    # filter out weak detections by requiring a minimum
                    # confidence
                    if confidence > configs['confidence']:
                        # extract the index of the class label from the
                        # detections list
                        idx = int(detections[0, 0, i, 1])

                        # if the class label is not a person, ignore it
                        if CLASSES[idx] != "person":
                            continue

                        # compute the (x, y)-coordinates of the bounding box
                        # for the object
                        box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                        (startX, startY, endX, endY) = box.astype("int")

                        # construct a dlib rectangle object from the bounding
                        # box coordinates and then start the dlib correlation
                        # tracker
                        tracker = dlib.correlation_tracker()
                        rect = dlib.rectangle(startX, startY, endX, endY)
                        tracker.start_track(rgb, rect)

                        # add the tracker to our list of trackers so we can
                        # utilize it during skip frames
                        trackers.append(tracker)

        # otherwise, we should utilize our object *trackers* rather than
        # object *detectors* to obtain a higher frame processing throughput
        else:
            # loop over the trackers
            for tracker in trackers:
                # set the status of our system to be 'tracking' rather
                # than 'waiting' or 'detecting'
                status = "Tracking"

                # update the tracker and grab the updated position
                tracker.update(rgb)
                pos = tracker.get_position()

                # unpack the position object
                startX = int(pos.left())
                startY = int(pos.top())
                endX = int(pos.right())
                endY = int(pos.bottom())

                # add the bounding box coordinates to the rectangles list
                rects.append((startX, startY, endX, endY))

        # draw a horizontal line in the center of the frame -- once an
        # object crosses this line we will determine whether they were
        # moving 'up' or 'down'
        cv2.line(frame, (0, H // 2), (W, H // 2), (0, 255, 255), 2)

        # use the centroid tracker to associate the (1) old object
        # centroids with (2) the newly computed object centroids
        objects = ct.update(rects)
        # loop over the tracked objects
        for (objectID, centroid) in objects.items():
            # check to see if a trackable object exists for the current
            # object ID
            to = trackableObjects.get(objectID, None)

            # if there is no existing trackable object, create one
            if to is None:
                to = TrackableObject(objectID, centroid)

            # otherwise, there is a trackable object so we can utilize it
            # to determine direction
            else:
                totalUpPrevious = totalUp
                totalDownPrevious = totalDown
                # the difference between the y-coordinate of the *current*
                # centroid and the mean of *previous* centroids will tell
                # us in which direction the object is moving (negative for
                # 'up' and positive for 'down')
                y = [c[1] for c in to.centroids]
                direction = centroid[1] - np.mean(y)
                to.centroids.append(centroid)
                # check to see if the object has been counted or not
                if not to.counted:
                    # if the direction is negative (indicating the object
                    # is moving up) AND the centroid is above the center
                    # line, count the object
                    if direction < 0 and centroid[1] < H // 2:
                        totalUp += 1
                        iotUp += 1
                        to.counted = True

                    # if the direction is positive (indicating the object
                    # is moving down) AND the centroid is below the
                    # center line, count the object
                    elif direction > 0 and centroid[1] > H // 2:
                        totalDown += 1
                        iotDown += 1
                        to.counted = True

            # store the trackable object in our dictionary
            trackableObjects[objectID] = to

            # draw both the ID of the object and the centroid of the
            # object on the output frame
            if (configs['captureVideo'] == 1) or (configs['captureImages'] == 1):
                text = "ID {}".format(objectID)
                cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

        # construct a tuple of information we will be displaying on the
        # frame
        info = [
            ("Up", totalUp),
            ("Down", totalDown),
            ("Status", status),
        ]

        # loop over the info tuples and draw them on our frame
        if (configs['captureVideo'] == 1) or (configs['captureImages'] == 1):
            for (i, (k, v)) in enumerate(info):
                text = "{}: {}".format(k, v)
                cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        if (totalUp > totalUpPrevious) or (totalDown > totalDownPrevious):
            pushFrame = imutils.resize(frame,width=40)
            imageBase64 = cv2.imencode('.png', pushFrame)[1]

            iotImage = base64.b64encode(imageBase64)
            msg = {}
            msg['message'] = "Count"
            msg['up'] = iotUp
            msg['down'] = iotDown
            msg['frameCount'] = iotFrameCnt
            msg['image'] = str(iotImage)
            jsonMsg = json.dumps(msg)
            myMQTTClient.publishAsync(
                "human/" + configs['iotDeviceName'] + "/count", jsonMsg, 0, ackCallback=iotResponse)
            iotFrameCnt = 0
            iotUp = 0
            iotDown = 0

        # Increment the previous to the current and then reuse it above
        totalUpPrevious = totalUp
        totalDownPrevious = totalDown

        # check to see if we should write the frame to disk
        if configs['captureVideo'] == 1:
            writer.write(frame)

        # show the output frame
        if configs['showScreen'] == 1:
            cv2.imshow("Frame", frame)

        # increment the total number of frames processed thus far and
        # then update the FPS counter
        totalFrames += 1
        fps.update()

except KeyboardInterrupt:
# stop the timer and display FPS information
	fps.stop()
msg = {}
msg["message"] = "streamState"
msg["state"] = "stopped"
jsonMsg = json.dumps(msg)
myMQTTClient.publishAsync(
    "human/" + configs['iotDeviceName'] + "/status", jsonMsg, 0, ackCallback=iotResponse)

# check to see if we need to release the video writer pointer
if writer is not None:
     writer.release()

print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# if we are using a video file, stop
if configs['useVideoStream'] == 0:
    fvs.release()
# otherwise, release the video file pointer
else:
    fvs.stop()

# close any open windows
cv2.destroyAllWindows()
msg = {}
msg['message'] = "Application"
msg['state'] = "stopped"
jsonMsg = json.dumps(msg)
myMQTTClient.publishAsync(
    "human/" + configs['iotDeviceName'] + "/status", jsonMsg, 0, ackCallback=iotResponse)
myMQTTClient.disconnect()
cv2.destroyAllWindows()
sys.exit()
