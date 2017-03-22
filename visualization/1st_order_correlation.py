"""
Rec-sys Embedding Rec-sys prototyping

__author__:
    charles@qileap

"""
import csv, os
import seaborn as sns
import matplotlib.pyplot as plt
import math
import collections

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def read_csv(file_path, index_dict,ignore_null=True, binary_utility=True, sort=False):
    user_id_col = index_dict['user_id_col']
    item_id_col = index_dict['item_id_col']
    quantity_col = index_dict['quantity_col']
    ignore = 1
    user_stats = {}
    item_stats = {}
    item_distribution = []
    with open(file_path) as csv_file:
        spamrader = csv.reader(csv_file, delimiter=',', quotechar='\"')
        for row in spamrader:
            if ignore > 0:
                ignore -= 1
                continue
            # item distribution
            user_id, item_id, quantity = row[user_id_col], row[item_id_col], int(row[quantity_col])
            if ignore_null and user_id == '':
                continue
            if binary_utility:
                quantity = 1
            item_distribution.append(item_id)
            # utility matrix
            if user_id != "":
                if user_id in user_stats.keys():
                    user_stats[user_id] += quantity
                else:
                    user_stats[user_id] = quantity
            if item_id in item_stats.keys():
                item_stats[item_id] += quantity
            else:
                item_stats[item_id] = quantity
    avg_user_activity = sum(user_stats.values())/len(user_stats)
    avg_item_utility = sum(item_stats.values())/len(item_stats)
    # avg_user_activity = sum(user_stats.values())
    # avg_item_utility = sum(item_stats.values())
    print("average item utility",avg_item_utility,'number items', len(item_stats))
    print('average user utility', avg_user_activity,'number users', len(user_stats))
    print('item density',avg_item_utility/len(item_stats))
    print('user density',avg_user_activity/len(user_stats))
    print('(item utility / user utility) ratio', avg_item_utility/avg_user_activity)
    avg_user_activity = 'average user utility:'+str(avg_user_activity)
    avg_item_utility = 'average item utility: ' + str(avg_item_utility)
    label = ' \n'.join([avg_user_activity, avg_item_utility])
    # convert item distribution to sorted int
    counts = collections.Counter(item_distribution)
    # unsorted
    # sorted
    if sort:
        frequency_list = counts.most_common(len(counts))
    else:
        frequency_list = counts.items()
    frequency_list = [x[1] for x in frequency_list]
    return user_stats, item_stats, frequency_list, label



def plot_utility(transactions, label, log_y=False, log_x=False):
    if log_y:
        # nice mass representation
        values = [math.acosh(x) for x in transactions if x > 0]
        # values = [math.log10(x[1]) for x in transactions if x[1] > 0]
        # future
        # values = [math.asinh(x) for x in transactions if x > 0]
    else:
        values = [x for x in transactions if x > 0]
    # ax = sns.countplot(values)
    sns.plt.plot(values, label=label)
    # ax = sns.distplot(values, label=label, norm_hist=False, bins=200)
    # sns.distplot(values, label=label, rug=True, hist=False)
if __name__ == "__main__":
    def config_plot(plot_path, title, show=False):
        plt.xlim(0)
        plt.autoscale()
        # plt.ylim(0)
        plt.legend()
        plt.xlabel('item')
        plt.ylabel('utility')
        if show:
            sns.plt.show()
        sns.plt.title(title)
        sns.plt.savefig(plot_path)
        sns.plt.cla()

    dash_board = [['gift', 'movielens100k'][0]]
    # dash_board = ['gift', 'music', 'microsoft']
    show_plot = True
    sort = True


    if 'gift' in dash_board:
        param_dict = {
            "user_id_col" : 0,
            'item_id_col' : 1,
            'quantity_col': 2,
            'plot_title': 'item utility'
        }
        file_path = os.path.join('..', 'online_shopping', 'transaction_unlabelled')
        user, item, item_dist, label = read_csv(file_path, param_dict, ignore_null=True, sort=sort)
        plot_utility(item_dist, label, log_y=False)
        figure_path = os.path.join('plots', 'gift_item_popularity')
        config_plot(figure_path, param_dict['plot_title'],show=show_plot)


    elif 'movielens100k' in dash_board:
        param_dict = {
            "user_id_col" : 0,
            'item_id_col' : 1,
            'quantity_col': 2,
            'plot_title': 'item utility'
        }
        file_path = os.path.join('..','movielens100k', 'transaction_unlabelled')
        user, item, item_dist, label = read_csv(file_path, param_dict, ignore_null=True, sort=sort)
        plot_utility(item_dist, label, log_y=False)
        figure_path = os.path.join('plots', 'store_item_popularity')
        config_plot(figure_path, param_dict['plot_title'],show=show_plot)



    if 'artificial' in dash_board:
        param_dict = {
            "user_id_col" : 0,
            'item_id_col' : 1,
            'quantity_col': 2,
            'plot_title': 'item popularity'
        }
        file_path = os.path.join('..', 'artificial', 'uniform_6000_300')
        file_path = os.path.join('..', 'artificial', 'exponential_33000_285')
        user, item, item_dist, label = read_csv(file_path, param_dict, ignore_null=True, sort=sort)
        plot_utility(item_dist, label, log_y=False)
        figure_path = os.path.join('plots', 'artificial_item_popularity')
        config_plot(figure_path, param_dict['plot_title'],show=show_plot)