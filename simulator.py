"""
Rec-sys Embedding Rec-sys prototyping

__author__:
    charles@qileap

"""
from scipy import sparse
from abc import ABCMeta, abstractmethod
import random
import math

def sample_gen(n, forbid):
    state = dict()
    track = dict()
    for (i, o) in enumerate(forbid):
        x = track.get(o, o)
        t = state.get(n-i-1, n-i-1)
        state[x] = t
        track[t] = x
        state.pop(n-i-1, None)
        track.pop(o, None)
    del track
    for remaining in range(n-len(forbid), 0, -1):
        i = random.randrange(remaining)
        yield state.get(i, i)
        state[i] = state.get(remaining - 1, remaining - 1)
        state.pop(remaining - 1, None)

def project(l1, low, high):
    OldMax = max(l1)
    OldMin = min(l1)
    NewMax = high
    NewMin = low
    OldRange = (OldMax - OldMin)
    NewRange = (NewMax - NewMin)
    if OldRange == 0:
        NewValue = [NewMin for _ in l1]
    else:
        NewValue = [(((x - OldMin) * NewRange) / OldRange) + NewMin for x in l1]
    return NewValue

class RandomGenerator(object):
    __metaclass__ = ABCMeta
    def __init__(self):
        pass

    @abstractmethod
    def rand_float(self, scalar):
        pass
    @abstractmethod
    def draw_sample(self, bound):
        pass

class Exponential(RandomGenerator):
    def __init__(self):
        RandomGenerator.__init__(self)

    def rand_float(self, scalar=1):
        return random.random() * scalar

    def draw_sample(self, lambd):
        return random.expovariate(lambd)

class ErdosRenyi(RandomGenerator):
    def __init__(self):
        RandomGenerator.__init__(self)

    def rand_float(self, scalar=1):
        return random.random() * scalar

    def draw_sample(self, bound, replacement=False):
        gen = sample_gen(bound, [])
        if replacement:
            return next(gen)
        else:
            return random.randint(0, bound)


class Simulator(object):
    def __init__(self, dist):
        self.dist = dist
        pass

    def draw_sample(self, size):
        return self.dist.draw_sample(size)

    def flip(self, weight):
        return 1 if self.dist.rand_float(1) > weight else 0

    def gen_transactions(self, n_u, n_i, density, output_path=None):
        # generate the rectangular
        # and randomize the rectangular

        num_samples = int(n_u * n_i * density)
        samples = []
        if isinstance(self.dist, Exponential):
            lambd = 3
            for _ in range(num_samples):
                samples.append(self.draw_sample(lambd))
            samples = project(samples, 0, 1)
            samples = [int(x * (n_i-1)) for x in samples]
            print(min(samples),max(samples))
        elif isinstance(self.dist, ErdosRenyi):
            for _ in range(num_samples):
                samples.append(int(self.dist.rand_float()*num_items))
        else:
            raise Exception('Unknown distribution:', type(self.dist))
        utility_matrix = sparse.lil_matrix((n_u, n_i), dtype=int)
        print('to be filled:', num_samples)
        counter = 0
        for u in range(n_u):
            item_set = set()
            endurence = n_i
            for _ in range(n_i):
                if len(samples) <= 0:
                    break
                if self.flip(1-density):
                    pointer = 0
                    while samples[pointer] in item_set:
                        endurence -= 1
                        if endurence < 0:
                            break
                        pointer = random.randint(0, len(samples)-1)
                        # print(samples)
                    if endurence < 0:
                        break
                    counter += 1
                    item_set.add(samples[pointer])
                    utility_matrix[u, samples.pop(pointer)] = 1
        print('filled in:', counter)
        if output_path is not None:
            output_stream = open(output_path, 'w')
            users,items = utility_matrix.nonzero()
            item_count = {}
            user_count = {}
            for user_id,item_id in zip(users,items):
                if item_id in item_count.keys():
                    item_count[item_id] += 1
                else:
                    item_count[item_id] = 1
                if user_id in user_count.keys():
                    user_count[user_id] += 1
                else:
                    user_count[user_id] = 1
                transaction = [user_id,item_id,utility_matrix[user_id,item_id]]
                transaction = [str(x) for x in transaction]
                output_stream.write(",".join(transaction)+'\n')
            output_stream.close()
        return utility_matrix



if __name__ == "__main__":
    import os
    cur_dir = os.path.dirname(__file__)
    project_root = os.path.join(cur_dir, '..')
    output_folder = os.path.join(project_root, 'charles', 'artificial')
    # e_r = ErdosRenyi()
    e_r = Exponential()
    if isinstance(e_r, ErdosRenyi):
        model_name = 'uniform'
    elif isinstance(e_r, Exponential):
        model_name = 'exponential'
    else:
        raise Exception('Unknown model:', type(e_r))
    sim = Simulator(e_r)
    records = []
    # for _ in range(10000):
    num_users = 500
    num_items = 200
    density = 0.03
    print('number users:', num_users, 'num_items:', num_items)
    output_file_path = os.path.join(output_folder, '_'.join([model_name, str(num_users), str(num_items)])) #, str(density).replace('.','_')
    print(output_file_path)
    transactions = sim.gen_transactions(num_users,num_items, density, output_file_path)
