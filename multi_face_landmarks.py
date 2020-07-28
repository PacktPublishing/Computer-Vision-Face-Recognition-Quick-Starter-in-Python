# -*- coding: utf-8 -*-
"""
@author: abhilash
"""

import face_recognition
from PIL import Image, ImageDraw

#load the image file
face_image = face_recognition.load_image_file('images/testing/trump-modi.jpg')

#get the face landmarks list
face_landmarks_list =  face_recognition.face_landmarks(face_image)

#print the face landmarks list
print(len(face_landmarks_list))

#convert the numpy array image into pil image object
pil_image = Image.fromarray(face_image)
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

#show the final image    
pil_image.show()

#save the image
pil_image.save("images/samples/multi_landmarks.jpg")



