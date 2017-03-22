from keras.layers import Input, Dense, regularizers
from keras.models import Model
import numpy as np
import random, gensim

def gen_data_set(data_path, vocab, embedding, ratio=0.9, sample_rate=2.0*10e-2):
    item2idx = dict(zip(vocab, range(len(vocab))))
    idx2item = dict(zip(range(len(vocab)), vocab))
    train_inputs = []
    train_labels = []
    validation_inputs = []
    validation_labels = []
    # coding
    data = open(data_path, 'r')
    for receipt in data:
        receipt = receipt.strip().split(' ')
        for i in range(len(receipt)):
            if random.random() <= sample_rate:
                x_input = receipt[i]
                if x_input not in vocab:
                    continue
                remain = receipt.copy()
                remain.pop(i)
                labels = np.zeros(len(vocab))
                history = []
                for x in remain:
                    try:
                        history.append(item2idx[x])
                    except KeyError as e:
                        continue

                labels[history] = 1
                if random.random() < ratio:
                    train_inputs.append(embedding[x_input].tolist())
                    train_labels.append(labels.tolist())
                else:
                    validation_inputs.append(embedding[x_input].tolist())
                    validation_labels.append(labels.tolist())
    return (train_inputs, train_labels), (validation_inputs, validation_labels)

def AutoEncoder(encoding_dim, x_train, y_train, x_validation, y_validation, activation_1, activation_2, loss_function, epoch=30, bias=True):
    # this is the size of our encoded representations
    # dimension = 784
    # this is our input placeholder
    input_dimension = len(x_train[0])
    output_dimension = len(y_train[0])
    input_matrix = Input(shape=(input_dimension,))
    # "encoded" is the encoded representation of the input
    print('activation:', activation_1, activation_2)
    print(loss_function)
    # encoded = Dense(encoding_dim, activity_regularizer=regularizers.l2(10e-4), activation=activation_1, bias=bias)(input_matrix)
    encoded = Dense(encoding_dim,  activation=activation_1, bias=bias)(input_matrix)
    decoded = Dense(output_dimension,  activation=activation_2, bias=bias)(encoded)

    autoencoder = Model(input=input_matrix, output=decoded)

    encoder = Model(input=input_matrix, output=encoded)

    encoded_input = Input(shape=(encoding_dim,))
    decoder_layer = autoencoder.layers[-1]

    decoder = Model(input=encoded_input, output=decoder_layer(encoded_input))
    # create the decoder model
    autoencoder.compile(optimizer='adadelta', loss=loss_function)
    autoencoder.fit(x_train, y_train,
                    nb_epoch=epoch,
                    batch_size=256,
                    shuffle=True,
                    validation_data=(x_validation, y_validation))

    return encoder, decoder

if __name__ == "__main__":
    comments = '''
        change the sample strategy, choose nearby words in the content window and try train a model base on the sampled words.
    '''
    print(comments)
    import os
    np.set_printoptions(threshold=np.nan)
    project_name = 'movielens100k'
    # project_name = 'online_shopping'
    # project_name = 'tafeng'
    cur_dir = os.path.dirname(__file__)
    epoch = 17
    data_file_name = ['tr_short'][0]
    test_file_name = ['te_short'][0]
    model_name = ['short530sg', 'short550sg', 'short570sg', 'short5100sg'][1]+'.txt'
    # model_name = ['short510cbow', 'short530cbow', 'short550cbow', 'short570cbow', 'short5100cbow'][0]+'.txt'
    embedded_dimension = 50
    top_n = 1
    activation_functions = ['linear', 'sigmoid', 'relu', 'softmax']
    objective_functions = ['binary_crossentropy', 'kullback_leibler_divergence', 'mae', 'mse', 'mean_absolute_percentage_error'][0]
    activation_function_1 = activation_functions[1]
    activation_function_2 = activation_functions[0]
    print(project_name, activation_function_1, activation_function_2, objective_functions, model_name, 'epoch', epoch, 'top', top_n)
    online_shopping_transaction_file = os.path.join(cur_dir, project_name, data_file_name)
    online_shopping_model_path = os.path.join(cur_dir, project_name, 'models', model_name)
    embedding = gensim.models.Word2Vec.load_word2vec_format(online_shopping_model_path, binary=False)
    vocab = embedding.vocab
    (train_inputs, train_labels), (validation_inputs, validation_labels) = gen_data_set(online_shopping_transaction_file, vocab, embedding, ratio=0.9, sample_rate=10e-1)

    prev = []
    for x in train_labels:
        print(np.where(np.array(x) != 0)[0])
    exit()


    item2idx = dict(zip(vocab, range(len(vocab))))
    idx2item = dict(zip(range(len(vocab)), vocab))

    encoder, decoder = AutoEncoder(embedded_dimension, train_inputs, train_labels, validation_inputs, validation_labels, activation_function_1, activation_function_2, objective_functions, epoch=epoch, bias=True)

    # evaluate the precision
    with open(os.path.join(cur_dir, project_name, test_file_name), 'r') as f:
        hit = 0.0
        counter = 0
        recall = 0.0
        for line in f:
            items = line.strip().split(' ')
            if len(items) < 2:
                continue
            # pick 1 random item as query in each receipt
            pick = random.randrange(0, len(items))
            hidden = [x for i,x in enumerate(items) if i!=pick]
            try:
                query = embedding[items[pick]]
                encoded_query = encoder.predict(np.array([query]))
                print(encoded_query)
                recommendation = decoder.predict(encoded_query)
                # ret = idx2item[np.argmax(recommendation[0])]
                result = np.ndarray.argsort(recommendation[0])[-top_n:][::-1]
                # print([recommendation[0][x] for x in result])
                ret = [idx2item[x] for x in result]
                print(np.argmax(recommendation[0]), items[pick])
                print(query, ret)
                # print(idx2item[pick], ret, set(ret).intersection(hidden))
                hit += len(set(ret).intersection(hidden))
                counter += top_n
                recall += len(hidden)
                # if ret in hidden:
                #     precision += 1
            except KeyError:
                continue
        precision = hit/counter
        print(hit/(counter * top_n))
        print(hit/recall)
    # encoder, decoder = OneLayerAutoEncoder(dimension, embedded_dimension, train_noisy, x_train, test_noisy, x_test, 'relu', 'sigmoid', 'binary_crossentropy', epoch=50)
    # encoder, decoder = OneLayerAutoEncoder(dimension, embedded_dimension, train_noisy, x_train, test_noisy, x_test, 'relu', 'sigmoid', 'kullback_leibler_divergence', epoch=50)

