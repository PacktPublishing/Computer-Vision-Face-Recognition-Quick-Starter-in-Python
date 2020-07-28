# -*- coding: utf-8 -*-
"""

@author: abhilash
"""
#importing the required libraries
import cv2
import face_recognition


image_to_recognize_path = 'images/testing/trump.jpg'

#loading the image to detect
original_image = cv2.imread(image_to_recognize_path)

#load the sample images and get the 128 face embeddings from them
modi_image = face_recognition.load_image_file('images/samples/modi.jpg')
modi_face_encodings = face_recognition.face_encodings(modi_image)[0]

trump_image = face_recognition.load_image_file('images/samples/trump.jpg')
trump_face_encodings = face_recognition.face_encodings(trump_image)[0]

#save the encodings and the corresponding labels in seperate arrays in the same order
known_face_encodings = [modi_face_encodings, trump_face_encodings]
known_face_names = ["Narendra Modi", "Donald Trump"]

#load the unknown image to recognize faces in it
image_to_recognize = face_recognition.load_image_file(image_to_recognize_path)
image_to_recognize_encodings = face_recognition.face_encodings(image_to_recognize)[0]

#find the face distance of image_to_recognize with the known samples
face_distances = face_recognition.face_distance(known_face_encodings, image_to_recognize_encodings)

#printing the face distance value and sample names
for i,face_distance in enumerate(face_distances):
    print("The calculated face distance is {:.2} against the sample {}".format(face_distance,known_face_names[i]))
    print("The matching percentage is {} against the sample {}".format(round(((1-float(face_distance))*100),2),known_face_names[i]))









