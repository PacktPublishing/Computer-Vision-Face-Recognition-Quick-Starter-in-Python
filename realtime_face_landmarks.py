# -*- coding: utf-8 -*-
"""
@author: abhilash
"""

import face_recognition
from PIL import Image, ImageDraw
import numpy as np
import cv2


#capture the video from default camera 
webcam_video_stream = cv2.VideoCapture(0)
#webcam_video_stream = cv2.VideoCapture('images/testing/modi.mp4')

#initialize the array variable to hold all face locations in the frame
all_face_locations = []

#loop through every frame in the video
while True:
    #get the current frame from the video stream as an image
    ret,current_frame = webcam_video_stream.read()

    #get the face landmarks list
    face_landmarks_list =  face_recognition.face_landmarks(current_frame)
    
    #print the face landmarks list
    #print(len(face_landmarks_list))
    
    #convert the numpy array image into pil image object
    pil_image = Image.fromarray(current_frame)
    #convert the pil image to draw object
    d = ImageDraw.Draw(pil_image)
    
    #loop through every face
    index=0
    while index < len(face_landmarks_list):
        # loop through face landmarks
        for face_landmarks in face_landmarks_list:
          
            
            #join each face landmark points
            d.line(face_landmarks['chin'],fill=(255,255,255), width=2)
            d.line(face_landmarks['left_eyebrow'],fill=(255,255,255), width=2)
            d.line(face_landmarks['right_eyebrow'],fill=(255,255,255), width=2)
            d.line(face_landmarks['nose_bridge'],fill=(255,255,255), width=2)
            d.line(face_landmarks['nose_tip'],fill=(255,255,255), width=2)
            d.line(face_landmarks['left_eye'],fill=(255,255,255), width=2)
            d.line(face_landmarks['right_eye'],fill=(255,255,255), width=2)
            d.line(face_landmarks['top_lip'],fill=(255,255,255), width=2)
            d.line(face_landmarks['bottom_lip'],fill=(255,255,255), width=2)
    
        index +=1
    
    #convert PIL image to RGB to show in opencv window    
    rgb_image = pil_image.convert('RGB') 
    rgb_open_cv_image = np.array(pil_image)
    
    # Convert RGB to BGR 
    bgr_open_cv_image = cv2.cvtColor(rgb_open_cv_image, cv2.COLOR_RGB2BGR)
    bgr_open_cv_image = bgr_open_cv_image[:, :, ::-1].copy()

    #showing the current face with rectangle drawn
    cv2.imshow("Webcam Video",bgr_open_cv_image)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#release the stream and cam
#close all opencv windows open
webcam_video_stream.release()
cv2.destroyAllWindows()       



