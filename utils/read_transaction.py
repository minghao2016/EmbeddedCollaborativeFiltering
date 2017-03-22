import csv
from scipy.sparse import lil_matrix
import numpy as np


def read_transaction(transaction_file_path):
    with open(transaction_file_path, 'rb') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in spamreader:
            print(', '.join(row))


def gen_utility_matrix(transaction_file, rating=False):
    user_set = set()
    item_set = set()
    transactions = []
    with open(transaction_file, 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in spamreader:
            user_id, item_id, quantity = row[0], row[1], row[2]
            if user_id == '':
                continue
            user_set.add(user_id)
            item_set.add(item_id)
            transactions.append((user_id, item_id, quantity))
    print('num users:', len(user_set), 'num items:', len(item_set))
    # build user-idx and item-idx mapping for utility matrix
    user2idx = dict(zip(user_set, range(len(user_set))))
    idx2user = dict(zip(range(len(user_set)), user_set))
    item2idx = dict(zip(item_set, range(len(item_set))))
    idx2item = dict(zip(range(len(item_set)), item_set))
    utility_matrix = lil_matrix((len(user_set), len(item_set)), dtype=np.int8)
    for transaction in transactions:
        user_id, item_id, quantity = transaction
        if rating:
            if quantity == '4':
                quantity = 1
            elif quantity == '4.5':
                quantity = 2
            elif quantity == '5':
                quantity = 3
        else:
            quantity = int(quantity)
        # quantity = int(quantity)
        utility_matrix[user2idx[user_id], item2idx[item_id]] += quantity
    utility_package = {
        'utility_matrix': utility_matrix.toarray(),
        'idx2user': idx2user,
        'user2idx': user2idx,
        'item2idx': item2idx,
        'idx2item': idx2item
    }
    return utility_package


def split_utility_matrix(utility_matrix, ratio):
    num_row, num_col = utility_matrix.shape
    train = num_row * ratio
    return np.split(utility_matrix, [int(train)])


if __name__ == "__main__":
    import os
    # printing setting
    np.set_printoptions(threshold=np.nan)

    cur_dir = os.path.dirname(__file__)
    transaction_file = os.path.join(cur_dir, 'toy_transaction')
    online_shopping_transaction_file = os.path.join(cur_dir, '..', 'online_shopping', 'labelled')
    utility_data = gen_utility_matrix(online_shopping_transaction_file)
    utility_matrix, idx2user, idx2item = utility_data['utility_matrix'], utility_data['idx2user'], utility_data['idx2item']
    print(utility_matrix.shape)
    print(type(utility_matrix))
    train, test = split_utility_matrix(utility_matrix, 0.9)
    print(train[0], len(train))
    print(len(test))
    # print(utility_matrix[0], idx2user[0])
