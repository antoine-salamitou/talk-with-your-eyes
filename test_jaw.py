import dlib
import cv2
import numpy as np
from math import hypot
from math import ceil
from threading import Timer
import time
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import string

cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
font = cv2.FONT_HERSHEY_PLAIN


def calculate_jaw_distance():
    _, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = detector(gray)

    if len(faces) > 0:

        x, y = faces[0].left(), faces[0].top()
        x1, y1 = faces[0].right(), faces[0].bottom()
        cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2)

        landmarks = predictor(gray, faces[0])

        jaw_mid = ((landmarks.part(3).x + landmarks.part(13).x) / 2, (landmarks.part(3).y + landmarks.part(13).y) /2)
        jaw_region = np.array([(landmarks.part(3).x, landmarks.part(3).y),
            (landmarks.part(4).x, landmarks.part(4).y),
            (landmarks.part(5).x, landmarks.part(5).y),
            (landmarks.part(6).x, landmarks.part(6).y),
            (landmarks.part(7).x, landmarks.part(7).y),
            (landmarks.part(8).x, landmarks.part(8).y),
            (landmarks.part(9).x, landmarks.part(9).y),
            (landmarks.part(10).x, landmarks.part(10).y),
            (landmarks.part(11).x, landmarks.part(11).y),
            (landmarks.part(12).x, landmarks.part(12).y),
            (landmarks.part(13).x, landmarks.part(13).y)
            ], np.int32)

        cv2.polylines(frame, [jaw_region], True, 255, 2)


        
 
        jaw_distance = hypot((jaw_mid[0] - landmarks.part(8).x),(jaw_mid[1] - landmarks.part(8).y))
        
        
        
        left_eye_mid = ((landmarks.part(18).x + landmarks.part(20).x) / 2, (landmarks.part(18).y + landmarks.part(20).y) /2)
        left_eye_mid2 = ((landmarks.part(2).x + landmarks.part(31).x) / 2, (landmarks.part(2).y + landmarks.part(31).y) /2)
        left_eye_distance = hypot((left_eye_mid[0] - left_eye_mid2[0]),(left_eye_mid[1] - left_eye_mid2[1]))

        right_eye_mid = ((landmarks.part(18).x + landmarks.part(20).x) / 2, (landmarks.part(18).y + landmarks.part(20).y) /2)
        right_eye_mid2 = ((landmarks.part(2).x + landmarks.part(31).x) / 2, (landmarks.part(2).y + landmarks.part(31).y) /2)
        right_eye_distance = hypot((right_eye_mid[0] - right_eye_mid2[0]),(right_eye_mid[1] - right_eye_mid2[1]))

        eye_distance = (left_eye_distance + right_eye_distance) / 2


        eyebrow_region_left = np.array([(landmarks.part(17).x, landmarks.part(17).y),
                (landmarks.part(18).x, landmarks.part(18).y),
                (landmarks.part(19).x, landmarks.part(19).y),
                (landmarks.part(20).x, landmarks.part(20).y),
                (landmarks.part(21).x, landmarks.part(21).y)
                ], np.int32)

        eyebrow_region_right = np.array([(landmarks.part(22).x, landmarks.part(22).y),
            (landmarks.part(23).x, landmarks.part(23).y),
            (landmarks.part(24).x, landmarks.part(24).y),
            (landmarks.part(25).x, landmarks.part(25).y),
            (landmarks.part(26).x, landmarks.part(26).y)
            ], np.int32)
        


        cv2.polylines(frame, [eyebrow_region_left], True, 255, 2)
        cv2.polylines(frame, [eyebrow_region_right], True, 255, 2)
        

        return frame, jaw_distance, eye_distance


jaw_distance_threshold = 140
eye_distance_threshold = 165
sentence = "Sentence : "
leftPartAlphabet = False
last_jaw_up = False
last_eye_up = False

alphabetList = list(string.ascii_uppercase)
lettersInterval = ""



def getLetterIntervals(new_step, leftPartAlphabet):
    global alphabetList
    global lettersInterval
    global sentence
    if leftPartAlphabet == True:
        lettersInterval = alphabetList[0] + "-" +  alphabetList[int(len(alphabetList) / 2) - 1]
        if (new_step == True):
            alphabetList = alphabetList[0:int(len(alphabetList) / 2)]
            
    else:
        
        if (new_step == True):
            
            alphabetList = alphabetList[int(len(alphabetList) / 2): len(alphabetList)]
            lettersInterval = alphabetList[0] + "-" +  alphabetList[int(len(alphabetList) / 2) - 1]
            
        else:
            lettersInterval = alphabetList[int(len(alphabetList) / 2)] + "-" +  alphabetList[len(alphabetList) - 1]
    if len(alphabetList) == 1:
        sentence += alphabetList[0]
        alphabetList = list(string.ascii_uppercase)

TimerName = None

def changeSensPartAlphabet():
    global TimerName
    global leftPartAlphabet
    if leftPartAlphabet == True:
        leftPartAlphabet = False
    else:
        leftPartAlphabet = True
        
    TimerName = Timer(4, changeSensPartAlphabet)
    TimerName.start()

TimerName = Timer(4, changeSensPartAlphabet)
TimerName.start()

def lastJawDown():
    global last_jaw_up
    last_jaw_up = False

def lastEyeDown():
    global last_eye_up
    last_eye_up = False


jaw_count = 0


while True:

        


        
        
        frame, jaw_distance, eye_distance = calculate_jaw_distance()

        print("jaw distance : ")
        print(jaw_distance)
        print("eye distance : ")
        print(eye_distance)
        getLetterIntervals(False, leftPartAlphabet)
        cv2.putText(frame, lettersInterval, (600, 100), font, 10, (0, 0, 255), 3)
        cv2.putText(frame, sentence, (100, 680), font, 4, (0, 0, 255), 3)
        cv2.imshow("Frame", frame)


        if eye_distance > eye_distance_threshold and last_eye_up == False:
            last_eye_up = True
            Timer(1, lastEyeDown).start()
            TimerName.cancel()
            changeSensPartAlphabet()
        
           

        if jaw_distance > jaw_distance_threshold:
            jaw_count += 1
        else:
            jaw_count = 0

        if (jaw_count > 10):
            sentence = sentence[:-1]
            jaw_count = jaw_count - 4
            alphabetList = list(string.ascii_uppercase)


        if jaw_distance > jaw_distance_threshold and last_jaw_up == False:
            getLetterIntervals(True, leftPartAlphabet)
            last_jaw_up = True
            Timer(1, lastJawDown).start()
            jaw_count += 1
            TimerName.cancel()
            TimerName = Timer(4, changeSensPartAlphabet)
            TimerName.start()
            if leftPartAlphabet == False:
                leftPartAlphabet = True
            
       



        key = cv2.waitKey(1)
        if key == 27:
            break


            
            

        
  


cap.release()
cv2.destroyAllWindows()