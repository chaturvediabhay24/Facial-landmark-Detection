import cv2
import numpy as np
import dlib
from curve_fitting import curve

# cap = cv2.VideoCapture(0)

path='./results'
img_path='face.jpeg'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")



frame = cv2.imread(img_path)
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

faces = detector(gray)
for face in faces:
        landmarks = predictor(gray, face)
        l=[]

        for i in range(5, 12):                          #loop for storing coordinates of jaw
            curr_cordi=(landmarks.part(i).x, landmarks.part(i).y)
            l.append(curr_cordi)

        cur=np.array(curve(np.array(l)), np.int32)      # calling function to find proper fitting curve


        for i in range(len(cur)-1):                          #loop for drawing jaw line
            curr_cordi=(cur[i][0], cur[i][1])
            next_cordi=(cur[i+1][0], cur[i+1][1])
            cv2.line(frame, curr_cordi, next_cordi, (0, 0, 255), 1)
        for n in range(0, 68):                          #loop for drawing landmarks on face
        	x = landmarks.part(n).x
        	y = landmarks.part(n).y
        	cv2.circle(frame, (x, y), 3, (255, 0, 0), -1)



cv2.imwrite(path+"/"+img_path, frame)
