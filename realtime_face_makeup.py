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
          
            
            #draw the shapes and fill with color 
            
            # Make left, right eyebrows darker 
            # Polygon on top and line on bottom with dark color
            d.polygon(face_landmarks['left_eyebrow'], fill=(68, 54, 39, 128))
            d.polygon(face_landmarks['right_eyebrow'], fill=(68, 54, 39, 128))
            d.line(face_landmarks['left_eyebrow'], fill=(68, 54, 39, 150), width=5)
            d.line(face_landmarks['right_eyebrow'], fill=(68, 54, 39, 150), width=5)
        
        
            # Add lipstick to top and bottom lips
            # using red polygons and lines filled with red
            d.polygon(face_landmarks['top_lip'], fill=(0, 0, 200, 100))
            d.polygon(face_landmarks['bottom_lip'], fill=(0, 0, 200, 100))
            d.line(face_landmarks['top_lip'], fill=(150, 150, 150, 64), width=2)
            d.line(face_landmarks['bottom_lip'], fill=(150, 150, 150, 64), width=2)
        
        
            # Make left and right eyes filled with green
            d.polygon(face_landmarks['left_eye'], fill=(0, 255, 0, 100))
            d.polygon(face_landmarks['right_eye'], fill=(0, 255, 0, 100))
        
            # Eyeliner to left and right eyes as lines
            d.line(face_landmarks['left_eye'] + [face_landmarks['left_eye'][0]], fill=(0, 0, 0, 110), width=1)
            d.line(face_landmarks['right_eye'] + [face_landmarks['right_eye'][0]], fill=(0, 0, 0, 110), width=1)

    
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



