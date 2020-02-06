import warnings
warnings.filterwarnings('ignore')
import os
import numpy as np
import cv2
from keras.models import load_model
from keras.preprocessing import image

model=load_model('Model/model.bin')

face_haar_cascade=cv2.CascadeClassifier('Cascade Classifier/haarcascade_frontalface_default.xml')

# model_shape=model.input_shape[1:3]

cap=cv2.VideoCapture('sample.mp4')

while True:
	# Capture Image And Returns a Bool value
	ret,test_img=cap.read()
	if not ret:
		continue
	# gray_img=cv2.cvtColor(test_img,cv2.COLOR_BGR2GRAY)

	faces_detected=face_haar_cascade.detectMultiScale(test_img,scaleFactor=1.3,minNeighbors=5)

	for(x,y,w,h) in faces_detected:
		cv2.rectangle(test_img,(x,y),(x+w,y+h),(255,0,0),thickness=2)
		roi_gray=test_img[y:y+w,x:x+h] # Cropping Face From The Frame Captured
		# print(roi_gray.shape)
		img=cv2.resize(roi_gray,(48,48))
		img=img.reshape(img.shape[0],img.shape[1],-1)
		img=img.reshape(-1,img.shape[0],img.shape[1],img.shape[2])

		max_index=np.argmax(model.predict(img))
		print(model.predict_classes(img))

		emotions=('Angry','Contempt','Disgust','Fear','Happy','Sad','Surprise')
		predicted_emotion=emotions[max_index]

		cv2.putText(test_img,predicted_emotion,(int(x),int(y)),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)

	resized_img=cv2.resize(test_img,(1000,700))
	cv2.imshow('FACIAL EMOTION DETECTION',resized_img)


	# Press q to exit
	if cv2.waitKey(10)==ord('q'):
		break
cap.release()
