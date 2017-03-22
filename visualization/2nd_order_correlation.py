"""
Rec-sys Embedding Rec-sys prototyping

__author__:
    charles@qileap

"""

from sklearn.cluster.bicluster import SpectralCoclustering
import math, os, gensim, scipy.sparse
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt

def sigmoid(x):
    return 1 / (1 + math.exp(-x))
ZERO = 0.0000000000000000000001
def gen_co_occurrence_matrix(transaction_file_path, schema, similarity=False, model=None, smoothing=False):
    utility_matrix = dict()
    item_set = set()
    user_set = set()
    with open(transaction_file_path) as f:
        for line in f:
            data = line.strip().split(',')
            user_id = data[schema['user_id_col']]
            item_id = data[schema['item_id_col']]
            quantity = data[schema['quantity_col']]
            item_set.add(item_id)
            user_set.add(user_id)
            if user_id in utility_matrix.keys():
                utility_matrix[user_id].append(item_id)
            else:
                utility_matrix[user_id] = [item_id]
    # build mapping
    item_id_to_index_mapping = {x:i for i,x in enumerate(item_set)}
    print(len(item_id_to_index_mapping))
    counter = 0
    if similarity:
        item_set = list(item_set)
        cooccurrence_matrix = np.zeros(shape=(len(item_set), len(item_set)))
        counter = 0
        for i in range(len(item_set)):
            for j in range(len(item_set)):
                counter += 1
                # print('%3f' % (counter / len(item_set) / len(item_set)))
                if i == j:
                    cooccurrence_matrix[i, j] = 0
                    continue
                try:
                    similarity = model.similarity(item_set[i], item_set[j])
                    cooccurrence_matrix[i, j] = similarity
                except KeyError:
                    cooccurrence_matrix[i, j] = 0
    else: # co-occurrence (distance) matrix

        cooccurrence_matrix = scipy.sparse.lil_matrix((len(item_set), len(item_set)))
        for user in utility_matrix.keys():

            counter += 1
            # print('\r%.3f' % (counter/len(list(utility_matrix.keys()))), len(utility_matrix[user]), end='')
            for i in range(len(utility_matrix[user])):

                this_item = utility_matrix[user][i]
                others = utility_matrix[user][:]
                del others[i]

                for other_item in others:
                    # addition
                    x = item_id_to_index_mapping[this_item]
                    y = item_id_to_index_mapping[other_item]
                    if not smoothing:
                        cooccurrence_matrix[x, y] += 1
                        # print(cooccurrence_matrix[x,y])
                    elif smoothing:
                        assert 1 == 2
                        # smoothing
                        # print(cooccurrence_matrix.maxprint)
        cooccurrence_matrix = cooccurrence_matrix.toarray()
    return cooccurrence_matrix

if __name__ == "__main__":

    cur_dir = os.path.dirname(__file__)
    project = ['online_shopping', 'artificial', 'microsoft', 'music','kaggle', 'store', 'bank_competition'][1]
    window_size = 30
    dimension = 50
    model_name = ['labelled', 'unlabelled', 'uniform_10000_1000'][2]
    file_path = os.path.join(cur_dir, '..', project, model_name)
    model_path = os.path.join(cur_dir, '..', project, 'models', model_name+str(window_size)+str(dimension)+'.txt')
    embedding = [False, True][1]
    rank = [False, True][1]
    smooth = [False, True][0]
    regenerate_model = [False, True][0]
    binary = [False, True][0]
    offset = [1][0]
    color_scheme = [plt.cm.Blues, plt.cm.Spectral_r][1]
    param_dict = {
        "user_id_col" : 0,
        'item_id_col' : 1,
        'quantity_col': 2,
        'plot_title': 'item popularity'
    }
    try:
        # for debugging
        # raise FileNotFoundError()

        if embedding:
            co_matrix_file_path = os.path.join(cur_dir, '..', project, '_'.join(['cooccurrent_matrix_embedding', str(window_size), str(dimension)]))
            if regenerate_model:
                os.remove(co_matrix_file_path)
            matrix_file = open(co_matrix_file_path, 'rb')
        else:
            co_matrix_file_path = os.path.join(cur_dir, '..', project, '_'.join(['cooccurrent_matrix']))
            if regenerate_model:
                os.remove(co_matrix_file_path)
            matrix_file = open(co_matrix_file_path, 'rb')
        co_matrix = np.loadtxt(matrix_file)
    except FileNotFoundError:
        print('building similarity matrix....', end='')
        if embedding:
            word2vec_encoder = gensim.models.Word2Vec.load_word2vec_format(model_path, binary=False)
            co_matrix = gen_co_occurrence_matrix(file_path, param_dict, similarity=embedding, model=word2vec_encoder, smoothing=smooth)
            matrix_file_path = os.path.join(cur_dir, '..', project, '_'.join(['cooccurrent_matrix_embedding', str(window_size), str(dimension)]))
        else:
            co_matrix = gen_co_occurrence_matrix(file_path, param_dict, smoothing=smooth)
            matrix_file_path = os.path.join(cur_dir, '..', project, '_'.join(['cooccurrent_matrix']))
        matrix_file = open(matrix_file_path, 'wb')
        np.savetxt(matrix_file, co_matrix)
        print('done')

    copy = co_matrix
    stats = []
    options = range(0, 200, 10)
    options = [x/100 for x in options]
    print(options)
    if embedding:
        # spectrum analysis
        for option_index in range(len(options)):
            co_matrix = copy.copy()
            if binary: # binary
                threshold = options [option_index]
                co_matrix[co_matrix >= threshold] = threshold
                co_matrix[co_matrix < threshold] = ZERO
                num_max = len(np.where(co_matrix == threshold)[0])
                stats.append(num_max/co_matrix.shape[0]*co_matrix.shape[1])
                title_stats = ' threshold: '+str(threshold)
            else:
                if option_index == len(options) - 1:
                    continue
                size = 1
                low_bound, upper_bound = options[option_index] , \
                                         options[option_index+size]
                co_matrix += offset
                print('min value, max value:',co_matrix.min(), co_matrix.max())
                print('lower_bound, upper_bound:', low_bound, upper_bound)
                co_matrix[co_matrix == 0] = ZERO
                num_between = len(co_matrix[(low_bound < co_matrix) & (co_matrix < upper_bound)])
                print('num between', num_between)
                stats.append(num_between)
                title_stats = ' range: ('+str(low_bound)+','+str(upper_bound)+')'
            model = SpectralCoclustering(n_clusters=5, )# random_state=5, )
            model.fit(co_matrix)
            fit_data = co_matrix[np.argsort(model.row_labels_)]
            fit_data = fit_data[:, np.argsort(model.column_labels_)]
    co_matrix = copy.copy()
    # min_value = 0
    max_value = co_matrix.max()
    if not embedding:
        sigmoid = np.vectorize(sigmoid, otypes=[np.float])
        co_matrix = sigmoid(co_matrix)
    co_matrix[co_matrix == 0] = ZERO
    max_value = co_matrix.max()
    min_value = co_matrix.min()

    print(min_value, max_value)
    model = SpectralCoclustering(n_clusters=50, )# random_state=5, )
    model.fit(co_matrix)
    np.set_printoptions(threshold=np.nan)
    fit_data = co_matrix[np.argsort(model.row_labels_)]
    fit_data = fit_data[:, np.argsort(model.column_labels_)]
    # model = SpectralBiclustering()
    # clustered plot
    # cax = plt.imshow(co_matrix, cmap=color_scheme)
    if not rank:
        cax = sns.plt.matshow(co_matrix, cmap=color_scheme)
    else:
        cax = sns.plt.matshow(fit_data, cmap=color_scheme)
    if min_value > 0:
        min_value = 0
    plt.colorbar(cax, ticks=[min_value, max_value])
    # plt.imshow(fit_data, cmap=color_scheme)
    plt.show()
    if embedding:
        # y = [int(x) for x in stats]
        y = [x for x in stats]
        print(y)
        # y.reverse()
        x = [float(i)-offset for i in options[1:]]
        sns.barplot(x, y)
        sns.plt.show()

