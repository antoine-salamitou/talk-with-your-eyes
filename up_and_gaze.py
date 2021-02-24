import dlib
import cv2
import numpy as np
from math import hypot
from threading import Timer
import time
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def midpoint(p1, p2):
    return int((p1.x + p2.x) / 2), int((p1.y + p2.y) /2)



def midpoint2(p1, p2):
    return int((p1[0] + p2[0]) / 2), int((p1[1] + p2[1]) /2)


def get_blinking_ratio(eye_points, facial_landmarks):
    
    left_point = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
    right_point = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)
    center_top = midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
    center_bottom = midpoint(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4]))

    #hor_line = cv2.line(frame, left_point, right_point, (0, 255, 0), 2)
    #ver_line = cv2.line(frame, center_top, center_bottom, (0, 255, 0), 2)

    hor_line_length = hypot((left_point[0] - right_point[0]),(left_point[1] - right_point[1]))
    ver_line_length = hypot((center_top[0] - center_bottom[0]),(center_top[1] - center_bottom[1]))
    ratio = hor_line_length / ver_line_length
    return ratio

def separate_eye(eye_points, facial_landmarks):
    eye_region = np.array([(facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y),
            (facial_landmarks.part(eye_points[1]).x, facial_landmarks.part(eye_points[1]).y),
            (facial_landmarks.part(eye_points[2]).x, facial_landmarks.part(eye_points[2]).y),
            (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y),
            (facial_landmarks.part(eye_points[4]).x, facial_landmarks.part(eye_points[4]).y),
            (facial_landmarks.part(eye_points[5]).x, facial_landmarks.part(eye_points[5]).y)], np.int32)

    height, width, _ = frame.shape
    mask = np.zeros((height, width), np.uint8)
    
    cv2.polylines(mask, [eye_region], True, 255, 2)
    cv2.polylines(frame, [eye_region], True, 255, 2)

    cv2.fillPoly(mask, [eye_region], 255)
    eye = cv2.bitwise_and(gray, gray, mask=mask)


    min_x = np.min(eye_region[:, 0])
    max_x = np.max(eye_region[:, 0])
    min_y = np.min(eye_region[:, 1])
    max_y = np.max(eye_region[:, 1])

    gray_eye = eye[min_y: max_y, min_x: max_x]
    return gray_eye

def get_gaze_ratio(eye_points, facial_landmarks):
    
    gray_eye = separate_eye(eye_points, facial_landmarks)
    _, threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY)
    height, width = threshold_eye.shape

    left_side_threshold = threshold_eye[0: height, 0: int(width/2)]
    left_side_white = cv2.countNonZero(left_side_threshold)
    
    right_side_threshold = threshold_eye[0: height, int(width/2) : width]
    right_side_white = cv2.countNonZero(right_side_threshold)
    if right_side_white == 0:
        gaze_ratio = 5
    else:
        gaze_ratio = left_side_white / right_side_white

    return gaze_ratio 



def get_up_value(eye_points, facial_landmarks):
    gray_eye = separate_eye(eye_points, facial_landmarks);


    _, threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY)
    height, width = threshold_eye.shape

    down_side_threshold = threshold_eye[0: int(height/2), 0: width]
    up_side_threshold = threshold_eye[int(height/2) : height, 0: width]
    down_side_white = cv2.countNonZero(down_side_threshold)
    up_side_white = cv2.countNonZero(up_side_threshold)

    if down_side_white == 0:
        up_ratio = 0
    else:
        up_ratio = up_side_white / down_side_white
    return up_ratio

blinking = 0
time_left = 0
time_right = 0
time_up = 0


last_blinking = False
last_left = False
last_right = False
last_up = False


def LastLeftToFalse():
    global last_left
    last_left = False

def LastRightToFalse():
    global last_right
    last_right = False

def LastBlinkToFalse():
    global last_blinking
    last_blinking = False

def LastUpToFalse():
    global last_up
    last_up = False



font = cv2.FONT_HERSHEY_PLAIN


SentenceToWrite = "Sentence : "
firstCount = False
secondCount = False
count1 = 0
count2 = 0
arrayChar = [["a","b","c","d", " "],["e","f","g","h"],["i","j","k","l","m"],["n","o","p","q","r","s"],["t","u","v","w","x","y","z"]]


blink_count = 0


blinking_ratio_threshold = 3.8  
up_value_threshold = 5
gaze_value_threshold_right = 0.25
gaze_value_threshold_left = 5



while True:
 
    _, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = detector(gray)

    for face in faces:
        x, y = face.left(), face.top()
        x1, y1 = face.right(), face.bottom()
        cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2)

        landmarks = predictor(gray, face)
        
        left_eye_ratio = get_blinking_ratio([36, 37, 38, 39, 40, 41], landmarks)
        right_eye_ratio = get_blinking_ratio([42, 43, 44, 45, 46, 47], landmarks)

        blinking_ratio = (left_eye_ratio +  right_eye_ratio) /2




        


        gaze_ratio_left_eye = get_gaze_ratio([36, 37, 38, 39, 40, 41], landmarks)
        gaze_ratio_right_eye = get_gaze_ratio([42, 43, 44, 45, 46, 47], landmarks)
        gaze_value_ratio = (gaze_ratio_left_eye+gaze_ratio_right_eye)/2
        


        up_value_left_eye = get_up_value([36, 37, 38, 39, 40, 41], landmarks)
        up_value_right_eye = get_up_value([42, 43, 44, 45, 46, 47], landmarks)
        up_value = (up_value_right_eye + up_value_left_eye) / 2
    
        print("gaze_ratio")
        print(gaze_value_ratio)
        print("up value")
        print(up_value) 
        print('time left')
        print(time_left)
        print('time_right')
        print(time_right)
        print('timeup')
        print(time_up)
        print('blinking ratio')
        print(blinking_ratio)
        
        if blinking_ratio > 4:
            blink_count += 1
        else:
            blink_count = 0

        if (blink_count > 10):
            SentenceToWrite = SentenceToWrite[:-1]
            blink_count = blink_count - 2
   
    


        #blinking
        if blinking_ratio > blinking_ratio_threshold and last_blinking == False and last_up == False and last_left == False and last_right == False :

            blinking += 1
            last_blinking = True
            Timer(0.2, LastBlinkToFalse).start()
            if secondCount == True:
                count2 +=1
            if firstCount == True and secondCount == False:
                count1 += 1

        
    
            

        #look left
        #if gaze_value_left_side > gaze_value_threshold_left and gaze_value_right_side <0.66 * gaze_value_left_side and last_left == False and last_blinking == False and last_right == False and last_up == False:
        if gaze_value_ratio > gaze_value_threshold_left and last_left == False and last_blinking == False and last_right == False and last_up == False and up_value < up_value_threshold:
            time_left += 1;
            last_left = True
            Timer(1, LastLeftToFalse).start()
            if count1 > 0 and count2 > 0:
                if(count1 > 5 or count2 > 7):
                    cv2.putText(frame, "ERROR START AGAIN", (50, 150), font, 3, (255, 0, 0))
                else:
                    SentenceToWrite += arrayChar[count1 - 1][count2 - 1]
                count1 = 0
                count2 = 0
                firstCount = False
                secondCount = False
            
            

        #look right
       # if gaze_value_right_side > gaze_value_threshold_right and gaze_value_left_side < 0.66 * gaze_value_right_side and last_right == False and last_blinking == False and last_left == False and last_up == False:
        if gaze_value_ratio < gaze_value_threshold_right and last_right == False and last_blinking == False and last_left == False and last_up == False and up_value < up_value_threshold:
            time_right += 1
            last_right = True
            Timer(1, LastRightToFalse).start()

            if secondCount == True:
                secondCount = False
                count2 = 0
            else:
                firstCount = False
                count1 = 0

        
        #look up
        #if up_value > up_value_threshold and last_up == False and last_blinking == False and gaze_value_left_side >  0.7 * gaze_value_threshold_left and  gaze_value_right_side > 0.7 * gaze_value_threshold_right:
        if up_value > up_value_threshold and last_up == False and last_blinking == False and last_right == False and last_left == False:
            time_up += 1
            last_up = True
            Timer(1, LastUpToFalse).start()

            if firstCount == False:
                firstCount = True
            elif secondCount == False and firstCount == True and count1 > 0:
                secondCount = True
        
        

            cv2.putText(frame, "ABORT START AGAIN", (50, 150), font, 3, (255, 0, 0))
        
        cv2.putText(frame, "Blinking : " + str(blinking), (600, 100), font, 2, (0, 0, 255), 3)
        cv2.putText(frame, SentenceToWrite, (100, 680), font, 4, (0, 0, 255), 3)
        cv2.putText(frame, "time right : "+ str(time_right), (50, 100), font, 2, (0, 0, 255), 3)
        cv2.putText(frame, "time left : " + str(time_left), (300, 100), font, 2, (0, 0, 255), 3)
        cv2.putText(frame, "time up : " + str(time_up), (900, 100), font, 2, (0, 0, 255), 3)



        cv2.putText(frame, "count1 : " + str(count1), (800, 300), font, 4, (0, 0, 255), 3)
        cv2.putText(frame, "count2 : " + str(count2), (800, 400), font, 4, (0, 0, 255), 3)
        cv2.putText(frame, "firstCount : " + str(firstCount), (100, 300), font, 4, (0, 0, 255), 3)
        cv2.putText(frame, "secondCount : " + str(secondCount), (100, 400), font, 4, (0, 0, 255), 3)



    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break


cap.release()
cv2.destroyAllWindows()