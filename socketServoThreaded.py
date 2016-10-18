from socketIO_client import SocketIO, BaseNamespace
import requests
import json
import time
import Queue
import threading
import sys
import RPi.GPIO as GPIO
from pyimagesearch.tempimage import TempImage
from dropbox.client import DropboxOAuth2FlowNoRedirect
from dropbox.client import DropboxClient
from picamera.array import PiRGBArray
from picamera import PiCamera
import argparse
import warnings
import datetime
import imutils
import cv2
import base64

exit_threads = 0
motion_detect = 0
message = {
    'user_id': 'testUser'
}

class PrivateNamespace(BaseNamespace):
    def on_connect( self ):
        print "connected."
class cameraThread(threading.Thread):
    def __init__(self, threadID, name,msgQueue,lock):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
	self.msgQueue = msgQueue
	self.lock = lock
        #self.socketIO = socketIO
        #self.user_nsp = user_nsp
        #self.timeout_signals = 0
        #self.msgQueue = msgQueue
        #self.lock = lock

    def run(self):
        # construct the argument parser and parse the arguments
        ap = argparse.ArgumentParser()
        ap.add_argument("-c", "--conf", required=True,
	help="path to the JSON configuration file")
        args = vars(ap.parse_args())

        # filter warnings, load the configuration and initialize the Dropbox
        # client
        warnings.filterwarnings("ignore")
        conf = json.load(open(args["conf"]))
        client = None

        # initialize the camera and grab a reference to the raw camera capture
        camera = PiCamera()
        camera.resolution = tuple(conf["resolution"])
        camera.framerate = conf["fps"]
        rawCapture = PiRGBArray(camera, size=tuple(conf["resolution"]))

        # allow the camera to warmup, then initialize the average frame, last
        # uploaded timestamp, and frame motion counter
        print "[INFO] warming up..."
        time.sleep(conf["camera_warmup_time"])
        avg = None
        lastUploaded = datetime.datetime.now()
        motionCounter = 0
	global motion_detect
        # capture frames from the camera
        for f in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
                # grab the raw NumPy array representing the image and initialize
                # the timestamp and occupied/unoccupied text
		if exit_threads:
		    break
		
                frame = f.array
                timestamp = datetime.datetime.now()
                text = "Unoccupied"

                # resize the frame, convert it to grayscale, and blur it
                frame = imutils.resize(frame, width=500)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (21, 21), 0)

                # if the average frame is None, initialize it
                if avg is None:
                        print "[INFO] starting background model..."
                        avg = gray.copy().astype("float")
                        rawCapture.truncate(0)
                        continue

                # accumulate the weighted average between the current frame and
                # previous frames, then compute the difference between the current
                # frame and running average
                cv2.accumulateWeighted(gray, avg, 0.5)
                frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(avg))

                # threshold the delta image, dilate the thresholded image to fill
                # in holes, then find contours on thresholded image
                thresh = cv2.threshold(frameDelta, conf["delta_thresh"], 255,
                        cv2.THRESH_BINARY)[1]
                thresh = cv2.dilate(thresh, None, iterations=2)

                cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cnts = cnts[0] if imutils.is_cv2() else cnts[1]
                #(cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                #	cv2.CHAIN_APPROX_SIMPLE)

	        # loop over the contours
        	for c in cnts:
                	# if the contour is too small, ignore it
                        if cv2.contourArea(c) < conf["min_area"]:
                                continue

                        # compute the bounding box for the contour, draw it on the frame,
                        # and update the text
                        (x, y, w, h) = cv2.boundingRect(c)
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        text = "Occupied"

                # draw the text and timestamp on the frame
                ts = timestamp.strftime("%A %d %B %Y %I:%M:%S%p")
                cv2.putText(frame, "Room Status: {}".format(text), (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.putText(frame, ts, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.35, (0, 0, 255), 1)


                # check to see if the room is occupied
                if True:
		#if text == "Occupied":
                        # check to see if enough time has passed between uploads
                        
                	if (timestamp - lastUploaded).seconds >= conf["min_upload_seconds"]:
                                # increment the motion counter
                        	motionCounter += 1

                                # check to see if the number of frames with consistent motion is
                                # high enough
                                
                                if text == "Occupied":
					print "motion was detected"
					self.lock.acquire()
					print "setting the motion flag"
                                        motion_detect = 1
					self.lock.release()
					# check to see if dropbox sohuld be used
                                        #if conf[True]:   #["use_dropbox"]:
                                                # write the image to temporary file
                        	

				t = TempImage()
                        	cv2.imwrite(t.path, frame)
                        	cnt = open(t.path, "rb").read()
                        	print type(cnt)
				b64 = base64.encodestring(cnt)
                        	self.lock.acquire()
				print "sending the stream"
				self.msgQueue.put(b64)
				self.lock.release()
                        	print type(b64)
            
                                        # upload the image to Dropbox and cleanup the tempory image
                        	print "[UPLOAD] {}".format(ts)
                                        #path = "{base_path}/{timestamp}.jpg".format(
                                        #	base_path=conf["dropbox_base_path"], timestamp=ts)
                                        #client.put_file(path, open(t.path, "rb"))
                        	t.cleanup()

                                        # update the last uploaded timestamp and reset the motion
                                        # counter
                        	lastUploaded = timestamp
                        	motionCounter = 0

                # otherwise, the room is not occupied
                else:
                        motionCounter = 0

                # check to see if the frames should be displayed to screen
                if conf["show_video"]:
                        # display the security feed
                        cv2.imshow("Security Feed", frame)
                        key = cv2.waitKey(1) & 0xFF

                        # if the `q` key is pressed, break from the lop
                        if key == ord("q"):
                            break

                # clear the stream in preparation for the next frame
                rawCapture.truncate(0)

        
class socketThread(threading.Thread):
    def __init__(self, threadID, name, socketIO,user_nsp,msgQueue,lock):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.socketIO = socketIO
        self.user_nsp = user_nsp
        self.timeout_signals = 0
        self.msgQueue = msgQueue
        self.lock = lock
    def onTimeout(self):
        print "timeout received"
        #self.socketIO.wait(seconds=5)
        #self.user_nsp.emit('timeoutCheck')

    def onMessage(self, *args):
        jd = args[0]
        self.lock.acquire()#get the shared lock and write message
        if ('message','stop') in jd.items():
            print "stop message received"
            self.msgQueue.put('stop')
        elif ('message','start') in jd.items():
            print "start message received"
            self.msgQueue.put('start')
        elif ('message','right') in jd.items():
            print "turning to the right"
            self.msgQueue.put('right')
        elif ('message','left') in jd.items():
            print "turning to the left"
            self.msgQueue.put('left')
        self.lock.release()

    def run(self):
        print "Starting " + self.name
        print "initializing socket events"
        self.user_nsp.on('timeoutCheck', self.onTimeout)
        self.user_nsp.emit('timeoutCheck')
        self.user_nsp.on('newMessage',self.onMessage)
        #self.user_nsp.emit('gameMessage',{'message':'stop'})
        self.user_nsp.emit('newMessage',{'message':'right'})
        global exit_threads
	global motion_detect
        while True:
	    self.lock.acquire()
	    if motion_detect:
		motion_detect = 0
		print "new motion message sent"
		self.user_nsp.emit("sendPushNotification",{"title":"Tess Detected an Anomaly","message":"We detected motion, please check your live feed."})	
	    
	    if not self.msgQueue.empty():
		imageString = self.msgQueue.get()
		print "new image in array"
		self.user_nsp.emit("newImage",{"image":True,"buffer":imageString})
	    self.lock.release()
            self.socketIO.wait(1.0 / 24)
            #self.user_nsp.emit('newMessage',{'message':'hello'})
            #self.timeout_signals += 1
            if exit_threads:
                print "exiting the socket thread"
                break

class servoThread(threading.Thread):
    def __init__(self, threadID, name,msgQueue,lock,servoMotor):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.msgQueue = msgQueue
        self.lock = lock
        self.curState = "on"
        self.nxtState = "on"
        self.curCycle = (160,7.5,160)#tuple representing the remaining cycles, the ,and the total cycles
        self.servoMotor = servoMotor
    def dutyCycle(self):
        global exit_threads
        cycles = self.curCycle[0]
        for i in range(cycles):
            self.lock.acquire()#check the message queue for alerts
            if not self.msgQueue.empty():
                self.nxtState = self.msgQueue.get()
                print self.nxtState
            self.lock.release()
            if self.nxtState != self.curState:#state change message was received
                if self.nxtState == "off":#should save the current location of the motor and return
                    print "stopping the motor"
            	    self.curState = "off"
            	    self.servoMotor.stop()
    		    GPIO.cleanup()
                    time.sleep(1)
                    return
                elif self.nxtState == "on":
                    print "starting the motor"
                    print "the start cycle value is "
                    print self.curCycle
                    GPIO.setmode(GPIO.BOARD)
                    GPIO.setup(7, GPIO.OUT)
                    self.servoMotor = GPIO.PWM(7, 50)
                    self.curState = "on"#set motor state to active again
    		    self.servoMotor.start(self.curCycle[1] +
                    ((self.curCycle[2] - self.curCycle[0])  * .03125) * (1 if self.curCycle[1] < 10 else -1))#restart motor to saved state
            elif self.curState == "off":
                return
            print "the value of cycle is " + str(self.curCycle[1] +
            ((self.curCycle[2] - self.curCycle[0])  * .03125) * (1 if self.curCycle[1] < 10 else -1))
            self.servoMotor.ChangeDutyCycle(self.curCycle[1] +
            ((self.curCycle[2] - self.curCycle[0])  * .03125) * (1 if self.curCycle[1] < 10 else -1))
            time.sleep(.0625)
            self.curCycle = (self.curCycle[0] - 1, self.curCycle[1], self.curCycle[2])#decrement the remaining cycles
        if self.curCycle[1] == 7.5:
            self.curCycle = (320, 12.5, 320)
        elif self.curCycle[1] == 12.5:
            self.curCycle = (160, 2.5, 160)
        else:
            self.curCycle = (160, 7.5, 160)

    def run(self):
        print "Starting thread " + self.name
        print "starting the servo"
        self.servoMotor.start(7.5)
        global exit_threads
        while True:
            if exit_threads:
                print "exiting the servo thread"
                if self.curState == "on":
                    self.servoMotor.stop()
                    GPIO.cleanup()
        	    break
            self.dutyCycle()

message = json.dumps(message)

url = 'http://uses.herokuapp.com/nspCreate'
base_url = 'http://uses.herokuapp.com'
headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
r = requests.post(url, headers=headers, data=message)
print "the response was " + r.content
print(r.content)
print type(r)
jd = r.json()
print type(jd)
if ('active_socket',True) in jd.items():#after receiving a verification that the channel is active, begin sending data
    print jd.items()
    socketIO = SocketIO(base_url, 80, PrivateNamespace)
    user_nsp = socketIO.define(PrivateNamespace, '/xcIlYoRfOCXEA6MlIPKFBHaqhay2')
    lock = threading.Lock()
    msgQueue = Queue.Queue(3)
    #msgQueue = Queue.Queue(3)
    #GPIO.setmode(GPIO.BOARD)
    #GPIO.setup(7, GPIO.OUT)
    #servoMotor = GPIO.PWM(7, 50)
    clientSocket = socketThread(1,"clientsocket",socketIO,user_nsp,msgQueue,lock)
    #servo = servoThread(2,"servo",msgQueue,lock,servoMotor)
    camera = cameraThread(3,"camera",msgQueue,lock)
    clientSocket.start()
    #servo.start()
    camera.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print "program interrupt, exiting the main thread"
        exit_threads = 1
