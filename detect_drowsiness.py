import cv2
import numpy as np
import dlib
from imutils import face_utils
from scipy.spatial import distance
import pygame


pygame.mixer.init()
pygame.mixer.music.load('/home/akhilesh/alarm1.wav')

EYE_AR_THRESH = 0.28
EYE_AR_CONT_FRAMES = 48
COUNTER = 0
ALARM_ON = False


face_cascade = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor/shape_predictor_68_face_landmarks.dat')

(ls, le) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
(rs, re) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']


# def soundalarm(path_to_directry_of_alram):
#     playsound(path_to_directry_of_alram)

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])

    eye_ratio= (A+B) / (2*C)
    return eye_ratio

cap = cv2.VideoCapture(1)

while(True):
    ret , frame = cap.read()
    frame = cv2.flip(frame, 1)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face_rect = face_cascade.detectMultiScale(gray_frame, 1.3 , 5)

     for x, y, w, h in face_rect:
         cv2.rectangle(frame, (x,y), (x+w, y+h), [0,255,0], 5)


    faces = detector(gray_frame,0)

    for face in faces:
        shape = predictor(gray_frame, face)
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[ls:le]
        rightEye = shape[rs:re]

        leftEyeAspectRatio = eye_aspect_ratio(leftEye)
        rightEyeAspectRatio = eye_aspect_ratio(rightEye)

        eyeAspectRatio = (leftEyeAspectRatio + rightEyeAspectRatio) / 2
        print(eyeAspectRatio)

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        #Detect if eye aspect ratio is less than threshold
        if(eyeAspectRatio < EYE_AR_THRESH):
            COUNTER += 1
            #If no. of frames is greater than threshold frames,
            if COUNTER >= EYE_AR_CONT_FRAMES:
                print(COUNTER)
                cv2.putText(frame, "You are Drowsy", (150,200), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,0,255), 2)
                if not ALARM_ON:
                    ALARM_ON = True
                    pygame.mixer.music.play(-1)
                    
        else:
            pygame.mixer.music.stop()
            ALARM_ON = False
            COUNTER = 0



    cv2.imshow('frame', frame)

    k = cv2.waitKey(10) & 0xff
    if k==27:
        break


cap.release()
cv2.destroyAllWindows()
