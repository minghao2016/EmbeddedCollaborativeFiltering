from keras.layers import Input, Dense
from keras.models import Model

dimension = 784

# this is the size of our encoded representations
encoding_dim = 36  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

# this is our input placeholder
input_img = Input(shape=(dimension,))
# "encoded" is the encoded representation of the input
encoded = Dense(encoding_dim, activation='relu')(input_img)
# "decoded" is the lossy reconstruction of the input
decoded = Dense(dimension, activation='sigmoid')(encoded)

# this model maps an input to its reconstruction
autoencoder = Model(input=input_img, output=decoded)

# this model maps an input to its encoded representation
encoder = Model(input=input_img, output=encoded)

# create a placeholder for an encoded (32-dimensional) input
encoded_input = Input(shape=(encoding_dim,))
# retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]
# create the decoder model
decoder = Model(input=encoded_input, output=decoder_layer(encoded_input))
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')




if __name__ == "__main__":
    import os
    import numpy as np
    from common.future.charles.utils.read_transaction import gen_utility_matrix, split_utility_matrix
    from common.future.charles.one_layer_autoencoder import Noising, visualize_result, visualize_result_sorted
    cur_dir = os.path.dirname(__file__)
    transaction_file = os.path.join(cur_dir, 'toy_transaction')
    # online_shopping_transaction_file = os.path.join(cur_dir, 'online_shopping', 'unlabelled')
    online_shopping_transaction_file = os.path.join(cur_dir, 'movielens100k', 'unlabelled_daily')
    np.set_printoptions(threshold=np.nan)
    utility_data = gen_utility_matrix(online_shopping_transaction_file)
    utility_matrix, idx2user, idx2item = utility_data['utility_matrix'], utility_data['idx2user'], utility_data[
        'idx2item']
    print(utility_matrix.shape)
    x_train, x_test = split_utility_matrix(utility_matrix, 0.9)
    train_noisy, test_noisy = Noising(x_train, x_test)


    from keras.datasets import mnist
    import numpy as np
    (x_train, _), (x_test, _) = mnist.load_data()
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))



    autoencoder.fit(x_train, x_train,
                    nb_epoch=50,
                    batch_size=256,
                    shuffle=True,
                    validation_data=(x_test, x_test))
    encoded_imgs = encoder.predict(x_test)
    decoded_imgs = decoder.predict(encoded_imgs)

    visualize_result_sorted(encoder, decoder, x_test, dimension, encoding_dim)
