from keras.layers import Input, Dense, regularizers
from keras.models import Model
from math import sqrt, log
from .utils.read_transaction import gen_utility_matrix, split_utility_matrix
import numpy as np
import tensorflow as tf



def OneLayerAutoEncoder(dimension, encoding_dim, x_train, y_train, x_test, y_test, activation_1, activation_2, loss_function, epoch=30):
    # this is the size of our encoded representations
    # dimension = 784
    # this is our input placeholder
    input_matrix = Input(shape=(dimension,))
    # "encoded" is the encoded representation of the input
    print('activation:', activation_1, activation_2)
    encoded = Dense(encoding_dim, activity_regularizer=regularizers.l2(10e-5), activation=activation_1, bias=True)(input_matrix)
    # encoded = Dense(encoding_dim,  activation=activation_1, bias=True)(input_matrix)
    decoded = Dense(dimension,  activation=activation_2, bias=True)(encoded)

    autoencoder = Model(input=input_matrix, output=decoded)

    encoder = Model(input=input_matrix, output=encoded)

    encoded_input = Input(shape=(encoding_dim,))
    decoder_layer = autoencoder.layers[-1]

    decoder = Model(input=encoded_input, output=decoder_layer(encoded_input))
    # create the decoder model
    print(loss_function)
    autoencoder.compile(optimizer='adadelta', loss=loss_function)
    autoencoder.fit(x_train, y_train,
                    nb_epoch=epoch,
                    batch_size=256,
                    shuffle=True,
                    validation_data=(x_test, y_test))

    return encoder, decoder

def visualize_result(encoder, decoder, test, input_dim, embedded_dim):
    # encode and decode some digits
    # note that we take them from the *test* set
    dim_x = int(sqrt(input_dim))
    print(input_dim, dim_x)

    encoded_imgs = encoder.predict(test)

    decoded_imgs = decoder.predict(encoded_imgs)
    # from sklearn.metrics.pairwise import cosine_similarity
    # this = encoded_imgs[0]
    # for other in encoded_imgs[1:]:
    #    print(cosine_similarity(this.reshape(1, -1), other.reshape(1, -1)))
    print(test.shape)
    print(decoded_imgs.shape)
    print(type(decoded_imgs), type(decoded_imgs[0]))

    import matplotlib.pyplot as plt

    n = 10  # how many digits we will display
    plt.figure(figsize=(30, 4))
    avg = []
    for i in range(n):
        # display original
        ax = plt.subplot(3, n, i + 1)
        # plt.imshow(test[i].reshape(dim_x, dim_x))
        plt.imshow(test[i].reshape(dim_x, dim_x))
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
        plt.imshow(decoded_imgs[i].reshape(dim_x, dim_x))

        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    print(avg)
    plt.show()

def visualize_result_sorted(encoder, decoder, test, input_dim, embedded_dim):
    # encode and decode some digits
    # note that we take them from the *test* set
    dim_x = int(sqrt(input_dim))
    print(input_dim, dim_x)

    encoded_imgs = encoder.predict(test)

    decoded_imgs = decoder.predict(encoded_imgs)
    # from sklearn.metrics.pairwise import cosine_similarity
    # this = encoded_imgs[0]
    # for other in encoded_imgs[1:]:
    #    print(cosine_similarity(this.reshape(1, -1), other.reshape(1, -1)))
    print(test.shape)
    print(decoded_imgs.shape)
    print(type(decoded_imgs), type(decoded_imgs[0]))

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


def Noising(x_train, x_test):
    noise_factor = 0.5
    x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
    x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)

    x_train_noisy = np.clip(x_train_noisy, 0., 1.)
    x_test_noisy = np.clip(x_test_noisy, 0., 1.)
    return x_train_noisy, x_test_noisy

if __name__ == "__main__":
    import os
    np.set_printoptions(threshold=np.nan)

    # cur_dir = os.path.dirname(__file__)
    # transaction_file = os.path.join(cur_dir, 'toy_transaction')
    # # online_shopping_transaction_file = os.path.join(cur_dir, 'online_shopping', 'unlabelled')
    # # online_shopping_transaction_file = os.path.join(cur_dir, 'movielens100k', 'unlabelled_daily')
    # # utility_data = gen_utility_matrix(online_shopping_transaction_file)
    #
    # online_shopping_transaction_file = os.path.join(cur_dir, 'movielens10m', 'labelled')
    # utility_data = gen_utility_matrix(online_shopping_transaction_file, rating=True)
    #
    #
    # utility_matrix, idx2user, idx2item = utility_data['utility_matrix'], utility_data['idx2user'], utility_data[
    #     'idx2item']
    # x_train, x_test = split_utility_matrix(utility_matrix, 0.9)



    from keras.datasets import mnist
    import numpy as np
    (x_train, _), (x_test, _) = mnist.load_data()
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

    x_train = x_train[:1500]
    x_test = x_test[:300]

    train_noisy, test_noisy = Noising(x_train, x_test)

    embedded_dimension = 100
    # print(utility_matrix.shape)
    num_samples, dimension = x_train.shape
    print('training set', x_train.shape)
    print('testing set', x_test.shape)
    encoder, decoder = OneLayerAutoEncoder(dimension, embedded_dimension, train_noisy, x_train, test_noisy, x_test, 'linear', 'linear', 'binary_crossentropy', epoch=10)
    # encoder, decoder = OneLayerAutoEncoder(dimension, embedded_dimension, train_noisy, x_train, test_noisy, x_test, 'relu', 'sigmoid', 'binary_crossentropy', epoch=50)
    # encoder, decoder = OneLayerAutoEncoder(dimension, embedded_dimension, train_noisy, x_train, test_noisy, x_test, 'relu', 'sigmoid', 'kullback_leibler_divergence', epoch=50)

    visualize_result_sorted(encoder, decoder, x_test, dimension, embedded_dimension)
