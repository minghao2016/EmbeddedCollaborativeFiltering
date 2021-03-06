"""
Rec-sys Embedding Rec-sys prototyping

__author__:
    charles@qileap

"""
import datetime
import os

import numpy

from Word2Vec import HybridModel, iiCF, uuCF
from utils.helper import Train, GenTrainAndTest

cur_dir = os.path.dirname(__file__)
project_root = os.path.join(cur_dir, '..')


if __name__ == "__main__":
    print('''
        Try user-user embedding, embed users to a embedded space.
        After embedding use user-user collaborative filtering on the embedded space.
        1. Find most similar users
        2. Weighted average or linear regression of the k most similar users.
        3. User the regenerated value for recommendation
    ''')
    # best configuration for gift store data is window_size 5 dimension 30 or 50

    # seed = 0  # the random is useless here because the collection.count() has its own random
    # random.seed(seed)
    # https://archive.ics.uci.edu/ml/datasets/Online+Retail
    project = 'online_shopping'
    # project = 'gift_receipt'
    # project = 'microsoft'
    # project = 'tafeng'
    project = 'movielens100k'
    # project = 'artificial'
    # artificial data are generated by simulator module
    labelled_data = ['short', 'long'][0]
    unlabelled_data = ['short', 'long'][0]
    test_file_name = ['te_short', 'te_long'][0]
    test_file_path = os.path.join(cur_dir, project, test_file_name)
    control_1 = 0
    scalar = [0.5, 1, 1.5, 2, 2.5, 3][3] # 0.5 for sg, 2 for cbow
    dimension = [40, 70][0]
    top_n = 1
    num_item_in_query = 1
    num_bins = 5
    transaction_folder = os.path.join(cur_dir, project)

    # parameters
    sample = [True, False][0]
    gen_data = [True, False][1]
    cbow = [True, False][0]
    train = [True, False][control_1]
    hs = [0,1][0]
    # alpha = False   # 1 -> short term 0 -> long term    False -> without utility  True -> with utility
    alpha = 1   # 1 -> short term 0 -> long term    False -> without utility  True -> with utility
    if train:
        epoch = 1
    else:
        epoch = 50

    '''
    movie lens sg
    10% remain 0.6

    '''
    # long term model
    long_window_size = 10
    long_term_input_file_name = 'tr_'+labelled_data
    long_term_model_name = labelled_data+str(long_window_size)+str(dimension)+('cbow' if cbow else 'sg')+str(scalar).replace('.','')
    transaction_folder = os.path.join(cur_dir, project)
    long_term_input_file_path = os.path.join(cur_dir, project, long_term_input_file_name)
    long_term_model_path = os.path.join(cur_dir, project, 'models', long_term_model_name+'.txt')

    # short term model
    short_window_size = 5
    short_term_input_file_name = 'tr_'+unlabelled_data
    short_term_model_name = unlabelled_data+str(short_window_size)+str(dimension)+('cbow' if cbow else 'sg')+str(scalar).replace('.','')
    short_term_input_file_path = os.path.join(cur_dir, project, short_term_input_file_name)
    short_term_model_path = os.path.join(cur_dir, project, 'models', short_term_model_name+'.txt')

    print('project:', project, 'cbow' if cbow else 'sg', 'epochs:', epoch, 'short term model', short_term_model_name, 'test file', test_file_name, 'scalar', scalar)
    print('long term window', long_window_size, 'short term window', short_window_size, 'embedded dimension', dimension)
    benchmarking_result = [list() for _ in range(epoch)]
    print('alpha:', alpha, 'query_length', num_item_in_query, 'topn', top_n)
    results = [list() for _ in range(1, num_bins)] #list of list

    input_file_path = 'cleaned'
    if gen_data:
        if ('movielens' in project) or ('microsoft' in project):
            # movie lens
            GenTrainAndTest(transaction_folder, input_file_path, ratio=0.9, resolution='weekly', time_stamp_format=None, ignore_first_row=False)
        elif 'tafeng' in project:
            GenTrainAndTest(transaction_folder, input_file_path, ratio=0.9, resolution='daily', time_stamp_format='%Y-%m-%d %H:%M:%S', ignore_first_row=True, delimiter=',')
        else:
            # others
            GenTrainAndTest(transaction_folder, input_file_path, ratio=0.9, resolution='daily', time_stamp_format='%m/%d/%Y %H:%M', ignore_first_row=True)

    for _ in range(epoch):
        if train:
            Train(short_term_input_file_path, short_term_model_path, scalar=scalar,window=short_window_size, dim=dimension, sample=sample, use_gensim=True, workers=8, cbow=cbow, hs=hs, model_name=short_term_model_name)
            # Train(long_term_input_file_path, long_term_model_path, scalar=scalar,window=long_window_size, dim=dimension, sample=sample, use_gensim=True, workers=8, cbow=cbow, hs=hs)
        print('{0}\r'.format(str(_ * 100 / epoch)[:4] + '%'), end='')
        for bin_num in range(1, num_bins):
            start = datetime.datetime.now()
            result = uuCF(short_term_model_path, test_file_path, ratio=bin_num/num_bins, topn=top_n, alpha=alpha, shuffle=True, divider=num_item_in_query, remove_already_purchased=True)
            # result = HybridModel(short_term_model_path, short_term_input_file_path, test_file_path, ratio=bin_num/num_bins, topn=top_n, alpha=alpha, shuffle=True, divider=num_item_in_query, remove_already_purchased=True)
            # result = iiCF(short_term_model_path, test_file_path, ratio=bin_num/num_bins, topn=top_n, with_utility=alpha, shuffle=True, divider=num_item_in_query)
            end = datetime.datetime.now()
            benchmarking_result[_].append(end-start)
            results[bin_num - 1].append(result)
    results = [ (numpy.average(x), numpy.std(x)) for x in results]
    benchmarking_result = [sum(x, datetime.timedelta()) / epoch for x in benchmarking_result]
    for i in range(len(results)):
        avg, std = results[i]
        print((i+1)/num_bins, avg)
