# # save the final model to file
# from tensorflow.keras.datasets import mnist
# from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D
# from tensorflow.keras.layers import MaxPooling2D
# from tensorflow.keras.layers import Dense
# from tensorflow.keras.layers import Flatten
# from tensorflow.keras.optimizers import SGD
#
# # load train and test dataset
# def load_dataset():
# 	# load dataset
# 	(trainX, trainY), (testX, testY) = mnist.load_data()
# 	# reshape dataset to have a single channel
# 	trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
# 	testX = testX.reshape((testX.shape[0], 28, 28, 1))
# 	# one hot encode target values
# 	trainY = to_categorical(trainY)
# 	testY = to_categorical(testY)
# 	return trainX, trainY, testX, testY
#
# # scale pixels
# def prep_pixels(train, test):
# 	# convert from integers to floats
# 	train_norm = train.astype('float32')
# 	test_norm = test.astype('float32')
# 	# normalize to range 0-1
# 	train_norm = train_norm / 255.0
# 	test_norm = test_norm / 255.0
# 	# return normalized images
# 	return train_norm, test_norm
#
# # define cnn model
# def define_model():
# 	model = Sequential()
# 	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
#--- 7x7
# 	model.add(MaxPooling2D((2, 2)))
#--- batchnorm
# 	model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
#--- batchnorm ??? - 3x3 ---> 5x5
# 	model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
# 	model.add(MaxPooling2D((2, 2)))
#--- batchnorm
# 	model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform'))
#--- batchnorm
# 	model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform'))
# 	model.add(MaxPooling2D((2, 2)))
#--- batchnorm
# 	model.add(Flatten())
# 	model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
# 	model.add(Dense(10, activation='softmax'))
# 	# compile model
# 	opt = SGD(learning_rate=0.01, momentum=0.9)
# 	model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
# 	return model
#
# # run the test harness for evaluating a model
# def run_test_harness():
# 	# load dataset
# 	trainX, trainY, testX, testY = load_dataset()
# 	# prepare pixel data
# 	trainX, testX = prep_pixels(trainX, testX)
# 	# define model
# 	model = define_model()
# 	# fit model
# 	model.fit(trainX, trainY, epochs=10, batch_size=32, verbose=0)
# 	# save model
# 	model.save('final_model.h5')
#
# # entry point, run the test harness
# run_test_harness()

# make a prediction for a new image.
from numpy import argmax
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from keras.models import load_model

# load and prepare the image
def load_image(filename):
	# load the image
	img = load_img(filename, grayscale=True, target_size=(28, 28))
	# convert to array
	img = img_to_array(img)
	# reshape into a single sample with 1 channel
	img = img.reshape(1, 28, 28, 1)
	# prepare pixel data
	img = img.astype('float32')
	img = img / 255.0
	return img

# load an image and predict the class
def run_example():
	# load the image
	img = load_image('62.png')
	# load model
	model = load_model('final_model.h5')
	# predict the class
	predict_value = model.predict(img)
	digit = argmax(predict_value)
	print("Digit is: " , digit)

# entry point, run the example
run_example()



