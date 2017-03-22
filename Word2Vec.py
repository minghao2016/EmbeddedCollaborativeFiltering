"""
Rec-sys Embedding Rec-sys prototyping

__author__:
    charles@qileap

"""
import gensim, random, os, datetime, collections


def GetNeighbors(vertices, model, topn=5):
    ret_neighbors = []
    if len(vertices) == 0:
        return ret_neighbors
    while True:
        try:
            ret_neighbors = model.most_similar(vertices, topn=topn*len(vertices))
            # ret_neighbors = [x[0] for x in ret_neighbors]
            break
        except KeyError as e:
            for i in range(len(vertices)):
                if vertices[i] in str(e):
                    vertices.remove(vertices[i])
                    break
            if len(vertices) == 0:
                return ret_neighbors
            continue

    return ret_neighbors

def iiCF(model_path, test_file_path, ratio=0.9, topn=3, with_utility=False, shuffle=True, divider=None,
         iterate_all=False, n_neighbours=5):
    # load models
    model = gensim.models.Word2Vec.load_word2vec_format(model_path, binary=False)
    precision = 0.0
    counter = 0.0
    hidden_ratio_divider = divider
    # load test file
    with open(test_file_path, 'r') as f:
        for line in f:
            items = line.strip().split(' ')
            # merge all items, convert user profile into user utility vector
            items = collections.Counter(items)
            items = dict(items)
            utility = list(items.values())
            items = list(items.keys())
            indices = list(range(len(items)))
            # shuffle the test receipt
            if shuffle:
                random.shuffle(indices)
            # divide the test receipt by hidden ratio
            if hidden_ratio_divider is None:
                divider = int(float(len(indices)) * ratio)
            else:
                divider = hidden_ratio_divider

            # query is remaining user profile
            query = indices[:divider]
            # removed is set for prediction
            removed = indices[divider:]
            removed = [items[x] for x in removed]
            # ignore current test if either removed or query is empty
            if len(removed) == 0 or len(query) == 0:
                continue
            # from this point, the test case is a valid test case, increment the counter by 1
            counter += 1
            neighbors = []

            if iterate_all:
                pass
            else:
                for item_index in query:
                    try:
                        ret = model.similar_by_word(items[item_index], topn=n_neighbours)
                    except KeyError:
                        continue
                    for each in ret:
                        # todo @charles, predict ratings for all items with items purchased by this user and sort the aggregation
                        # todo @charles, implement the full item collaborative filtering
                        if with_utility:
                            neighbors.append((each[0], each[1] * utility[item_index]))
                        else:
                            neighbors.append((each[0], each[1] ))
            # CHECK POINT

            # merge the short term and long term result
            result = {}
            for each in neighbors:
                # each[0] is item id, each[1] is the similarity score
                if each[0] not in result.keys():
                    result[each[0]] = each[1]
                else:
                    result[each[0]] += each[1]
            if len(result) == 0:
                continue
            # sort the items base on the aggregated similarity
            sorted_x = sorted(result.items(), key=lambda x:x[1], reverse=True)
            top_items = [x[0] for x in sorted_x]
            top_items = top_items[:topn]
            # convert the precision to top_n precision
            current_query_precision = float(len(set(top_items).intersection(set(removed)))) / float(topn)
            precision += current_query_precision
    # calculate the average precision
    try:
        precision =  (precision / counter)
    except ZeroDivisionError:
        # in case all test cases contain only 1 item
        precision = 0
    return precision


def uuCF(short_mem_model_path, test_file_path, ratio=0.9, topn=3, alpha=0.5, shuffle=True, divider=None,
         remove_already_purchased=True, n_neighbours=5):
    # load models
    short_mem_model = gensim.models.Word2Vec.load_word2vec_format(short_mem_model_path, binary=False)
    precision = 0.0
    counter = 0.0
    hidden_ratio_divider = divider
    # load test file
    with open(test_file_path, 'r') as f:
        for line in f:
            items = line.strip().split(' ')
            # merge all items, convert user profile into user utility vector
            if remove_already_purchased:
                items = list(set(items))
            # shuffle the test receipt
            if shuffle:
                random.shuffle(items)
            # divide the test receipt by hidden ratio
            if hidden_ratio_divider is None:
                divider = int(float(len(items)) * ratio)
            else:
                divider = hidden_ratio_divider

            # query is remaining user profile
            query = items[:divider]
            # removed is set for prediction
            removed = items[divider:]
            # ignore current test if either removed or query is empty
            if len(removed) == 0 or len(query) == 0:
                continue
            # from this point, the test case is a valid test case, increment the counter by 1
            counter += 1
            neighbors = []

            # get neighbours from short term memory model
            short_term_model_query = query
            recommendations = GetNeighbors(short_term_model_query, short_mem_model, topn=n_neighbours)
            for each in recommendations:
                neighbors.append((each[0], each[1] * alpha))

            # merge the short term and long term result
            result = {}
            for each in neighbors:
                # each[0] is item id, each[1] is the similarity score
                if each[0] not in result.keys():
                    result[each[0]] = each[1]
                else:
                    result[each[0]] += each[1]
            if len(result) == 0:
                continue
            # sort the items base on the aggregated similarity
            sorted_x = sorted(result.items(), key=lambda x:x[1], reverse=True)
            top_items = [x[0] for x in sorted_x]
            top_items = top_items[:topn]
            # convert the precision to top_n precision
            current_query_precision = float(len(set(top_items).intersection(set(removed)))) / float(topn)
            precision += current_query_precision
    # calculate the average precision
    try:
        precision =  (precision / counter)
    except ZeroDivisionError:
        # in case all test cases contain only 1 item
        precision = 0
    return precision


def HybridModel(short_mem_model_path, long_mem_model_path, test_file_path, ratio=0.9, topn=3, alpha=0.5, shuffle=True, divider=None, remove_already_purchased=False):
    # load models
    short_mem_model = gensim.models.Word2Vec.load_word2vec_format(short_mem_model_path, binary=False)
    long_mem_model = gensim.models.Word2Vec.load_word2vec_format(long_mem_model_path, binary=False)
    precision = 0.0
    counter = 0.0
    hidden_ratio_divider = divider
    # load test file
    with open(test_file_path, 'r') as f:
        for line in f:
            items = line.strip().split(' ')
            # merge all items, convert user profile into user utility vector
            if remove_already_purchased:
                items = list(set(items))
            # shuffle the test receipt
            if shuffle:
                random.shuffle(items)
            # divide the test receipt by hidden ratio
            if hidden_ratio_divider is None:
                divider = int(float(len(items)) * ratio)
            else:
                divider = hidden_ratio_divider

            # query is remaining user profile
            query = items[:divider]
            # removed is set for prediction
            removed = items[divider:]
            # ignore current test if either removed or query is empty
            if len(removed) == 0 or len(query) == 0:
                continue
            # from this point, the test case is a valid test case, increment the counter by 1
            counter += 1
            neighbors = []

            # get neighbours from short term memory model
            long_term_model_query = query
            short_term_model_query = query
            recommendations = GetNeighbors(short_term_model_query, short_mem_model, topn=topn)
            for each in recommendations:
                neighbors.append((each[0], each[1] * alpha))
            # get neighbours from long term memory model
            recommendations = GetNeighbors(long_term_model_query, long_mem_model, topn=topn)
            for each in recommendations:
                neighbors.append((each[0], each[1] * (1 - alpha)))

            # merge the short term and long term result
            result = {}
            for each in neighbors:
                # each[0] is item id, each[1] is the similarity score
                if each[0] not in result.keys():
                    result[each[0]] = each[1]
                else:
                    result[each[0]] += each[1]
            if len(result) == 0:
                continue
            # sort the items base on the aggregated similarity
            sorted_x = sorted(result.items(), key=lambda x:x[1], reverse=True)
            top_items = [x[0] for x in sorted_x]
            top_items = top_items[:topn]
            # convert the precision to top_n precision
            current_query_precision = float(len(set(top_items).intersection(set(removed)))) / float(topn)
            precision += current_query_precision
    # calculate the average precision
    try:
        precision =  (precision / counter)
    except ZeroDivisionError:
        # in case all test cases contain only 1 item
        precision = 0
    return precision

def TestAlpha(model_path, test_file_path, ratio=0.9, topn=3, pick=None, shuffle=True):

    word2vec_encoder = gensim.models.Word2Vec.load_word2vec_format(model_path, binary=False)
    # method 1 try all words and compute score of matched items
    total_score = 0.0
    counter = 0.0
    with open(test_file_path, 'r') as f:
        for line in f:
            items = line.strip().split(' ')
            if shuffle:
                random.shuffle(items)
            # try 10 , 3
            # print counter
            divider = int(float(len(items)) * ratio)
            remains = items[:divider]
            removed = items[divider:]
            if len(removed) == 0 or len(remains) == 0:
                continue
            counter += 1
            neighbors = []
            # print(len(remains))
            for each in remains:
                try:
                    answer = word2vec_encoder.similar_by_word(each, topn=5)
                    for pair in answer:
                        neighbors.append((pair[0], pair[1]))
                except KeyError:
                    continue
            # remains = GetNeighbors(remains, word2vec_encoder, topn=5)
            # for each in remains:
            #     neighbors.append((each[0], each[1]))
            # print(remains)
            result = {}
            for each in neighbors:
                if each[0] not in result.keys():
                    result[each[0]] = each[1]
                else:
                    result[each[0]] += each[1]
            # second item in tuple is similarity
            # order by second column
            # result = Counter(neighbors)

            if len(result) == 0:
                continue

            sorted_x = sorted(result.items(), key=lambda x:x[1], reverse=True)

            if topn == 1 and pick is not None:
                    if pick >= len(sorted_x):
                        top_items = [x[0] for x in [sorted_x[-1]]]
                    else:
                        top_items = [x[0] for x in [sorted_x[pick]]]
            elif topn >= 1:
                # top_items = [x[0] for x in sorted_x]
                top_items = [x[0] for x in sorted_x[:topn]]
            else:
                raise Exception('Recommendation has to be >',topn)
            total_score += float(len(set(top_items).intersection(set(removed)))) / float(topn)
    try:
        precision =  (total_score / counter)
    except ZeroDivisionError:
        precision = 0
    return precision

def Enrich(input_path, window, scalar, threshold=100, cbow=True):
    enriched_file_path = input_path+'{0}_{1}_{2}.enrich'.format(str(window), str(scalar).replace('.', ''), 'c' if cbow else 's')
    enrich_f = open(enriched_file_path, 'w')
    with open(input_path, 'r') as raw_file:
        for line in raw_file:
            items = line.strip().split(' ')
            diff = len(items) - window
            if scalar is None:
                # too expensive
                for _ in range(diff**2):
                    random.shuffle(items)
                    enrich_f.write(' '.join(items)+'\n')
            else:
                if threshold > diff > 0:
                    for _ in range(int(diff * scalar)):
                        random.shuffle(items)
                        enrich_f.write(' '.join(items)+'\n')
                else:
                    enrich_f.write(' '.join(items)+'\n')

    return enriched_file_path


class MySentences(object):
    def __init__(self, fname):
        self.fname = fname

    def __iter__(self):
        for line in open(os.path.join(self.fname)):
            yield line.split()

def Embedding(input_path, output_path, window, dim, cbow=True, binary=False, scalar=1, sample=True, workers=8, threshold=100, use_gensim=False, hs=0, model_name=''):
    if sample:
        print('Random sampling training samples...', end='', flush=True)
        input_path = Enrich(input_path, window, scalar, threshold=threshold, cbow=cbow)
        print('done.', flush=True)
    command = 'word2vec -train {0} -output {1} -debug 2 -size {2} -window {3} -sample 1e-4 -negative 5 -hs 0 -binary {5} -cbow {4} -threads {6}'.format(input_path, output_path, dim, window, '1' if cbow else '0', '1' if binary else '0', str(workers))
    if use_gensim:
        print('Building model using Gensim library...', end='', flush=True)
        sentences = MySentences(input_path)  # a memory-friendly iterator
        model = gensim.models.Word2Vec(sentences, size=dim, window=window,
                                       workers=workers, sg=(0 if cbow else 1),
                                       hs=hs)
        model.save_word2vec_format(output_path, binary=binary)
        print('done', flush=True)
    else:
        print(command)
        start = datetime.datetime.now()
        os.system(command)
        end = datetime.datetime.now()
        print(end-start)


if __name__ == "__main__":
    cur_dir = os.path.dirname(__file__)
    root = os.path.join(cur_dir, '..')
    input_file_path = os.path.join(root, 'data', 'online_shopping', 'train')
    output_file_path = os.path.join(root, 'models', 'embedding', 'online_shopping.txt')
    Embedding(input_file_path, output_file_path, 15, 50)