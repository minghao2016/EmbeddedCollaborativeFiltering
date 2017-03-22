import csv, os
from common.future.charles.utils.read_transaction import gen_utility_matrix

cur_dir = os.path.join(os.path.dirname(__file__))



if __name__ == "__main__":
    # l1 = ['1' ,'2', '3']
    # my_dict = dict(zip(l1, range(len(l1))))
    # inv_map = {v: k for k, v in my_dict.items()}
    # print(my_dict)
    # print(inv_map)
    file_path = os.path.join(cur_dir, 'toy_transaction')
    gen_utility_matrix(file_path)