import os
import facenet_recognition
import cv2
import shutil
import sys
import numpy as np
from matplotlib import pyplot as plt
from io import StringIO

class ListStream:
    def __init__(self):
        self.data = []
    def write(self, s):
        self.data.append(s)

def increase_brightness(img, value = 30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

def histEq(img):
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

    # equalize the histogram of the Y channel
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])

    # convert the YUV image back to RGB format
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    return img_output

# turn off stdout because facenet has so many messages...
old_stdout = sys.stdout
sys.stdout = mystdout = StringIO()

# declare threshold for accuracy
threshold = 0.4

# Check system to load directory
if os.name == "nt":
	slash = "\\" # windows
else:
	slash = "/" # unix

# declare data paths
dataDir = (".." + "%s" + "datasets" + "%s") % (slash, slash)

originalData = (dataDir + "%s" + "originals") % (slash)

trainData = (dataDir + "%s" + "train") % (slash)
trainDataAligned = (dataDir + "%s" + "train_aligned") % (slash)

testData = (dataDir + "%s" + "test") % (slash)
testDataAligned = (dataDir + "%s" + "test_aligned") % (slash)

capData = (dataDir + "%s" + "cap") % (slash)
capDataAligned = (dataDir + "%s" + "cap_aligned") % (slash)

# declare model and classifier path
modelDir = (".." + "%s" + "models" + "%s") % (slash, slash)
model = modelDir + "20180408-102900.pb"
classifier = modelDir + "classifier.pkl"
test_classifier_type = "svm" # type of model either svm or nn
weight = modelDir + "model_small.yaml" # local store weights

saved = ("%s" + "%s" + "cap" + "%s" + "%s") % (capData, slash, slash, "capture.jpg")

'''
# if aligned folders exist, delete it, and then recreate
if os.path.isdir(trainDataAligned):
	shutil.rmtree(trainDataAligned)
os.makedirs(trainDataAligned)
print(trainDataAligned + "created")

if os.path.isdir(testDataAligned):
	shutil.rmtree(testDataAligned)
os.makedirs(testDataAligned)
print(testDataAligned + "created")

if os.path.isdir(capDataAligned):
	shutil.rmtree(capDataAligned)
os.makedirs(capDataAligned)
print(capDataAligned + "created")


# align train dataset
facenet_recognition.align_input(trainData, trainDataAligned)
print("Train data aligned")

# train classifier from trainAligned
facenet_recognition.create_classifier(trainDataAligned, model, classifier, weight, test_classifier_type)

print("Classifier trained")

# align data in test
facenet_recognition.align_input(testData, testDataAligned)

#redirect output from testing classifier to variable
sys.stdout = output = ListStream()

# test classifier
facenet_recognition.test_classifier(
            testDataAligned, model, classifier, weight, test_classifier_type)


# parse last output -> [first, last, accuracy]
result = result = output.data[-4].split(" ")[-3:]
name = "%s_%s" % (result[0], result[1].replace(":", ""))

# turn on stdout
sys.stdout = old_stdout

print("%s %s" % (name, result[-1]))
'''

# take input from webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
while(cap.isOpened()):

    # display stream
    ret, frame = cap.read()
    bframe = increase_brightness(frame)
    cv2.imshow('Cam', bframe)

    # listen for keypress
    c = cv2.waitKey(1) % 256

    # check if face is detected first, maybe show bounding boxes and update per 3-5 frames

    # if 'c' is pressed
    if c == ord('c'):
            # delete from capDataAligned
            alignedCapPic = (capDataAligned + "%s" + "cap" + "%s" + "capture.png") % (slash, slash)
            if os.path.isfile(alignedCapPic):
                    os.remove(alignedCapPic)
            
            # save input in capData
            cv2.imwrite(saved, frame)

            # align captured frame from capData to capDataAligned
            facenet_recognition.align_input(capData, capDataAligned)

            # redirect output from testing classifier to a variable
            sys.stdout = output = ListStream()

            # check if an aligned picture was generated
            if os.path.isfile(alignedCapPic):

                    # test classifier with captured frame
                    facenet_recognition.test_classifier(
                            capDataAligned, model, classifier, weight, test_classifier_type)

                    # parse last output -> [first, last, accuracy]
                    result = output.data[-4].split(" ")[-3:]
                    
                    # turn on stdout to print out result
                    sys.stdout = old_stdout

                    # higher than threshold = open original image and window name -> person's name
                    # lower than certain threshold = "unknown" and do not open any image
                    if (float)(result[-1]) > threshold:
                            name = "%s_%s" % (result[0], result[1].replace(":", ""))
                            print(name)

                            # close window if already exists
                            if name != "":
                                    cv2.destroyWindow(name)

                            # get picture from db
                            im = cv2.imread((originalData + "%s" + "%s" + "%s" + "%s_0001.jpg") % (slash, name, slash, name), cv2.IMREAD_COLOR);
                            height, width, dim = im.shape
                            
                            maxHeight = 600
                            scale = maxHeight / height
                            
                            newHeight = (int)(height * scale)
                            newWidth = (int)(width * scale)

                            # show picture
                            cv2.namedWindow(name, cv2.WINDOW_NORMAL)
                            cv2.resizeWindow(name, newWidth, newHeight)
                            cv2.imshow(name, im)
                    else:
                            print("Unknown")

    # 'q' = exit
    elif c == ord('q'):
        break

    sys.stdout = mystdout = StringIO()

# release resources
cap.release()
cv2.destroyAllWindows()

print("Exit")
