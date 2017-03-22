from keras.layers import Input, Dense, regularizers
from keras.models import Model
import keras
from math import sqrt, log
from utils.read_transaction import gen_utility_matrix, split_utility_matrix
import numpy as np
import tensorflow as tf


def StackedAutoEncoder(dimension, encoding_dim, x_train, y_train, x_test, y_test, activation_1, activation_2, loss_function, epoch=30, factor=4, bias=False):
	# this is the size of our encoded representations

	input_matrix = Input(shape=(dimension,))
	encoded = Dense(128*factor, activation=activation_1, activity_regularizer=regularizers.l2(10e-4),bias=bias)(input_matrix)
	encoded = Dense(64*factor, activation=activation_1,bias=bias)(encoded)
	encoded = Dense(encoding_dim, activation=activation_1,bias=bias)(encoded)

	decoded = Dense(64*factor, activation=activation_1,bias=bias)(encoded)
	decoded = Dense(128*factor, activation=activation_1, bias=bias)(decoded)
	# decoded = Dense(dimension, activation=activation_2, bias=bias)(decoded)
	decoded = Dense(dimension, activation=activation_2, activity_regularizer=regularizers.l2(10e-4), bias=bias)(decoded)

	autoencoder = Model(input=input_matrix, output=decoded)
	# encoder = Model(input=input_matrix, output=encoded)


	# create a placeholder for an encoded (32-dimensional) input
	encoded_input_1 = Input(shape=(encoding_dim,))
	encoded_input_2 = Input(shape=(64*factor,))
	encoded_input_3 = Input(shape=(128*factor,))


	encoder_layer_1 = autoencoder.layers[-6]
	encoder_layer_2 = autoencoder.layers[-5]
	encoder_layer_3 = autoencoder.layers[-4]
	decoder_layer_1 = autoencoder.layers[-3]
	decoder_layer_2 = autoencoder.layers[-2]
	decoder_layer_3 = autoencoder.layers[-1]
	encoder_1 = Model(input=input_matrix, output=encoder_layer_1(input_matrix))
	encoder_2 = Model(input=encoded_input_3, output=encoder_layer_2(encoded_input_3))
	encoder_3 = Model(input=encoded_input_2, output=encoder_layer_3(encoded_input_2))


	# create the decoder model
	decoder_1 = Model(input = encoded_input_1, output = decoder_layer_1(encoded_input_1))
	decoder_2 = Model(input = encoded_input_2, output = decoder_layer_2(encoded_input_2))
	decoder_3 = Model(input = encoded_input_3, output = decoder_layer_3(encoded_input_3))
	# optimizer = keras.optimizers.Adagrad(lr=0.01, epsilon=1e-08, decay=0.0)
	# optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
	# optimizer = keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0)
	optimizer = keras.optimizers.Adadelta()
	autoencoder.compile(optimizer=optimizer, loss=loss_function)

	autoencoder.fit(x_train, y_train,
	                nb_epoch= epoch,
	                batch_size=40,
	                shuffle=True,
	                validation_data=(x_test, y_test))

	# return encoder, [decoder_1, decoder_2, decoder_3]
	return [encoder_1, encoder_2, encoder_3], [decoder_1, decoder_2, decoder_3]


def visualize_result(encoder, decoder, test, input_dim, embedded_dim):
	# print(dir(encoder[0]))
	# print(encoder[0].get_weights())
	# exit()
	# encode and decode some digits
	# note that we take them from the *test* set
	dim_x = int(sqrt(input_dim))
	print(input_dim, dim_x)

	# encoded_imgs = encoder.predict(test)
	encoded_imgs = encoder[0].predict(test)
	encoded_imgs = encoder[1].predict(encoded_imgs)
	encoded_imgs = encoder[2].predict(encoded_imgs)

	from sklearn.metrics.pairwise import cosine_similarity
	this = encoded_imgs[0]
	for other in encoded_imgs[1:]:
		print(cosine_similarity(this.reshape(1, -1), other.reshape(1, -1)))
	decoded_imgs = decoder[0].predict(encoded_imgs)
	decoded_imgs = decoder[1].predict(decoded_imgs)
	decoded_imgs = decoder[2].predict(decoded_imgs)
	print(test.shape)
	print(decoded_imgs.shape)
	print(type(decoded_imgs), type(decoded_imgs[0]))
	# exit()

	# use Matplotlib (don't ask)
	import matplotlib.pyplot as plt

	n = 10  # how many digits we will display
	plt.figure(figsize=(30, 4))
	avg = []
	for i in range(n):
		# display original
		ax = plt.subplot(3, n, i + 1)
		# plt.imshow(test[i].reshape(dim_x, dim_x))
		indices = test[i].argsort()
		original_matrix = test[i][indices[::-1]]
		original_matrix = np.resize(original_matrix, (dim_x, dim_x))
		plt.imshow(original_matrix)
		plt.gray()
		ax.get_xaxis().set_visible(False)
		ax.get_yaxis().set_visible(False)

		ax = plt.subplot(3, n, i + 1 + n)
		plt.imshow(encoded_imgs[i].reshape(int(sqrt(embedded_dim)), int(sqrt(embedded_dim))))
		# plt.gray()
		ax.get_xaxis().set_visible(False)
		ax.get_yaxis().set_visible(False)

		# display reconstruction
		ax = plt.subplot(3, n, i + 1 + 2 * n)
		# threshold the data
		# threshold = np.average(decoded_imgs[i]) + np.std(decoded_imgs[i])
		# threshold = 0.9
		# decoded_imgs[i][decoded_imgs[i]>threshold] = 1
		# decoded_imgs[i][decoded_imgs[i]<=threshold] = 0
		avg.append(np.average(decoded_imgs[i]))
		# print(test[i])
		# print(decoded_imgs[i], max(decoded_imgs[i]))
		# exit()
		# plt.imshow(decoded_imgs[i].reshape(dim_x, dim_x))
		decoded_matrix = decoded_imgs[i][indices[::-1]]
		decoded_matrix = np.resize(decoded_matrix, (dim_x, dim_x))
		plt.imshow(decoded_matrix)

		plt.gray()
		ax.get_xaxis().set_visible(False)
		ax.get_yaxis().set_visible(False)
	print(avg)
	plt.show()


def Noising(x_train, x_test, noise_factor):
	x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
	x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)
	# todo @charles the clip -1 to 1 or 0 to 1??
	x_train_noisy = np.clip(x_train_noisy, 0., 1.)
	x_test_noisy = np.clip(x_test_noisy, 0., 1.)
	return x_train_noisy, x_test_noisy

def Corrupt(x_train, corrupt_factor):
	x_train_corrupt = x_train

if __name__ == "__main__":
	import os

	cur_dir = os.path.dirname(__file__)
	transaction_file = os.path.join(cur_dir, 'toy_transaction')
	# online_shopping_transaction_file = os.path.join(cur_dir, 'online_shopping', 'unlabelled')

	online_shopping_transaction_file = os.path.join(cur_dir, 'movielens100k', 'labeled100k')
	online_shopping_transaction_file = os.path.join(cur_dir, 'movielens100k', 'unlabeled_weekly')
	# online_shopping_transaction_file = os.path.join(cur_dir, 'movielens100k', 'unlabeled')
	np.set_printoptions(threshold=np.nan)
	utility_data = gen_utility_matrix(online_shopping_transaction_file)
	utility_matrix, idx2user, idx2item = utility_data['utility_matrix'], utility_data['idx2user'], utility_data[
		'idx2item']
	print(utility_matrix.shape)
	x_train, x_test = split_utility_matrix(utility_matrix, 0.9)



	# from keras.datasets import mnist
	# import numpy as np
	# (x_train, _), (x_test, _) = mnist.load_data()
	# x_train = x_train.astype('float32') / 255.
	# x_test = x_test.astype('float32') / 255.
	# x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
	# x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))




	train_noisy, test_noisy = Noising(x_train, x_test, noise_factor=0.9)
	num_samples, dimension = x_train.shape
	embedded_dimension = 64
	print('training set', x_train.shape)
	print('testing set', x_test.shape)
	# encoder, decoder = StackedAutoEncoder(dimension, embedded_dimension, train_noisy, x_train, test_noisy, x_test, 'linear', 'linear', 'binary_crossentropy', epoch=20)
	# kullback_leibler_divergence
	encoder, decoder = StackedAutoEncoder(dimension, embedded_dimension, train_noisy, x_train, test_noisy, x_test, 'linear', 'sigmoid', 'binary_crossentropy', epoch=50, factor=1, bias=True)
	# visualize_result(encoder, decoder, x_test, dimension, embedded_dimension)
	visualize_result(encoder, decoder, x_train, dimension, embedded_dimension)