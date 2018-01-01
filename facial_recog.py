import numpy as np
import cv2
from model import FacialExpressionModel

rgb = cv2.VideoCapture(0)
facec = cv2.CascadeClassificer('haarcascade_frontalface_default.xml')
font = cv2.FONT_HERSHEY_SIMPLEX

def __get_data__():

	"""
	__get_data__: get data from the VideoCapture object and
	classifies them to a face or not a face.

	Return: tuple(faces in image, frame read, grayscale frame)
	"""

	_, fr = rgb.read()
	gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
	# Detects objects of different sizes in the input image. The detected objects are returned as a list of rectangles.
	faces = facec.detectMultiScale(gray, 1.3, 5)

	return faces, fr, gray


def start_app(facial_model):
	skip_frame = 10
	data = []
	flag = False
	ix = 0
	while True:
		ix += 1

		faces, fr, gray_fr = __get_data__()
		for x, y, w, h in faces:
			fc = gray_fr[y:y+h, x:x+w]

			roi = cv2.resize(fc, (48, 48))
			pred = facial_model.predict_emotion(roi[np.newaxis, :, :, np.newaxis])

			cv2.putText(fr, pred, (x, y), font, 1, (255, 255, 0), 2)
			cv2.rectangle(fr, (x, y), (x + w, y + h), (255, 0, 0), 2)

		if cv2.waitKey(1) == 27:
			break
		cv2.imshow('Filter', fr)
	cv2.destroyAllWindows()


if __name__ == '__main__':
	model = FacialExpressionModel('face_model.json', 'face_model.h5')
	start_app(model)
