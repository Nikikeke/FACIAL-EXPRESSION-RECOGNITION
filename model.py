import numpy as np
from keras.models import model_from_json

class FacialExpressionModel:

	EMOTION = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

	def __init__(self, model_json_file, model_weights_file):
		# load model
		with open(model_json_file, 'r') as json_file:
			loaded_model_json = json_file.read()
			self.loaded_model = model_from_json(loaded_model_json)

		# load weights
		self.loaded_model.load_weights(model_weights_file)
		print('Model loaded from file')
		self.loaded_model.summary()

	def predict_emotion(self, img):
		self.preds = self.loaded_model.predict(img)
		return FacialExpressionModel.EMOTION[np.argmax(self.preds)]

if __name__ == '__main__':
	pass