"""
Rec-sys Embedding Rec-sys prototyping

__author__:
    charles@qileap

"""
import os, random, datetime, csv


from Word2Vec import Embedding


def ConvertReceiptsToTransaction(file, output_file):
    output_stream = open(output_file, 'w')
    counter = 0
    with open(file, 'r') as f:
        for line in f:
            counter +=1
            play_list = line.strip().split(' ')
            user_id = counter
            for item in play_list:
                output_stream.write(', '.join([str(user_id), item, '1'])+'\n')
    output_stream.close()

# ConvertReceiptsToTransaction('store/retail.dat', 'store/unlabelled')

def ConvertMusicListToTransaction(file, output_file):
    output_stream = open(output_file, 'w')
    with open(file, 'r') as f:
        for line in f:
            line_buffer = line.strip().split(' ')
            user_id = line_buffer[0]
            play_list = line_buffer[1:]
            play_list = [x for x in play_list if ':' not in x]
            for song in play_list:
                output_stream.write(', '.join([user_id, song, '1'])+'\n')
    output_stream.close()

# ConvertMusicListToTransaction('music/music_list.txt', 'music/labelled')


def Encode(input_code):
    output = []
    input_code = input_code.replace(' ', '')
    for char in input_code:
        if char.isalpha():
            output.append(str(ord(char)))
        else:
            output.append(char)
    output = ''.join(output)
    # use this temp encoding for testing
    # output = ''.join([str(ord(x)%9) for x in input_code])
    return output


def GenTrainAndTest(folder_path, file_name, ratio=0.9, verbose=False, resolution='daily', time_stamp_format='%Y%m%d', ignore_first_row=False, delimiter=',', quotechar='"'):
    ret = {}
    transaction_file_path = os.path.join(folder_path, file_name)
    print(transaction_file_path)



    with open(transaction_file_path, 'r') as f:
        reader = csv.reader(f, delimiter=',', quotechar='"')
        if ignore_first_row:
            next(reader)
        for line in reader:
            user_id, item_id, quantity, time_stamp = line
            if user_id == '' or float(quantity)<0:
                continue
            if time_stamp_format is not None:
                time_stamp = datetime.datetime.strptime(time_stamp, time_stamp_format)
            elif time_stamp_format is None:
                time_stamp = datetime.datetime.fromtimestamp(int(time_stamp))
            if resolution == 'daily':
                time_stamp = time_stamp.strftime('%Y%m%d')
            elif resolution == 'weekly':
                # time_stamp.isocalendar() returns a 3-tuple, (ISO year, ISO week number, ISO weekday)
                week_number = time_stamp.isocalendar()[1]
                time_stamp = time_stamp.strftime('%Y')+str(week_number)
            elif resolution == 'monthly':
                time_stamp = time_stamp.strftime('%Y%m')
            elif resolution == 'yearly':
                time_stamp = time_stamp.strftime('%Y')
            else:
                raise Exception('unknown resolution option:', resolution, 'supported resolution options: daily, weekly, monthly, yearly')
            item_id = Encode(item_id)
            if user_id in ret.keys():
                ret[user_id].append((item_id, time_stamp, quantity))
            else:
                ret[user_id] = [(item_id, time_stamp, quantity)]

    long_train_receipts = []
    short_train_receipts = []
    test_long_receipts = []
    test_short_receipts = []
    transaction_labelled = os.path.join(folder_path, 'transaction_labelled')
    transaction_unlabelled = os.path.join(folder_path, 'transaction_unlabelled')
    transaction_train_unlabelled = os.path.join(folder_path, 'transaction_train_unlabelled')
    transaction_train_labelled = os.path.join(folder_path, 'transaction_train_labelled')
    transaction_test_unlabelled = os.path.join(folder_path, 'transaction_test_unlabelled')
    transaction_test_labelled = os.path.join(folder_path, 'transaction_test_labelled')
    transaction_labelled_writer = open(transaction_labelled, 'w')
    transaction_unlabelled_writer = open(transaction_unlabelled, 'w')
    transaction_train_unlabelled_writer = open(transaction_train_unlabelled, 'w')
    transaction_train_labelled_writer = open(transaction_train_labelled, 'w')
    transaction_test_unlabelled_writer = open(transaction_test_unlabelled, 'w')
    transaction_test_labelled_writer = open(transaction_test_labelled, 'w')
    transaction_labelled_writer_csv = csv.writer(transaction_labelled_writer, delimiter=delimiter, quotechar=quotechar)
    transaction_unlabelled_writer_csv = csv.writer(transaction_unlabelled_writer, delimiter=delimiter,
                                                   quotechar=quotechar)
    transaction_train_unlabelled_writer_csv = csv.writer(transaction_train_unlabelled_writer, delimiter=delimiter,
                                                         quotechar=quotechar)
    transaction_train_labelled_writer_csv = csv.writer(transaction_train_labelled_writer, delimiter=delimiter,
                                                       quotechar=quotechar)
    transaction_test_unlabelled_writer_csv = csv.writer(transaction_test_unlabelled_writer, delimiter=delimiter,
                                                        quotechar=quotechar)
    transaction_test_labelled_writer_csv = csv.writer(transaction_test_labelled_writer, delimiter=delimiter,
                                                      quotechar=quotechar)
    for user_id in ret.keys():
        # decompose the long term into short term
        short_ret = {}
        for item in ret[user_id]:
            item_id, ts, quantity = item
            tmp_uid = user_id + ts
            if tmp_uid in short_ret.keys():
                short_ret[tmp_uid].append((item_id, ts, quantity))
            else:
                short_ret[tmp_uid] = [(item_id, ts, quantity)]
            # write to labelled and unlabelled file
            transaction_labelled_writer_csv.writerow([user_id, item_id, quantity])
            transaction_unlabelled_writer_csv.writerow([tmp_uid, item_id, quantity])

        if random.random() < ratio:
            # append the item to the long_term_profile
            for tmp_uid in short_ret.keys():
                short_receipt = []
                for x in short_ret[tmp_uid]:
                    item_id, ts, quantity = x
                    transaction_train_unlabelled_writer_csv.writerow([tmp_uid, item_id, quantity])
                    short_receipt.append(item_id)
                short_train_receipts.append(short_receipt)
            long_receipt = []
            for x in ret[user_id]:
                item_id, ts, quantity = x
                long_receipt.append(item_id)
                transaction_train_labelled_writer_csv.writerow([user_id, item_id, quantity])
            long_train_receipts.append(long_receipt)

        else:
            # append the item to the long_term_profile
            for tmp_uid in short_ret.keys():
                short_receipt = []
                for x in short_ret[tmp_uid]:
                    item_id, ts, quantity = x
                    transaction_test_unlabelled_writer_csv.writerow([tmp_uid, item_id, quantity])
                    short_receipt.append(item_id)
                test_short_receipts.append(short_receipt)
            long_receipt = []
            for x in ret[user_id]:
                item_id, ts, quantity = x
                long_receipt.append(item_id)
                transaction_test_labelled_writer_csv.writerow([user_id, item_id, quantity])
            test_long_receipts.append(long_receipt)




    total = .0
    count = 0
    for receipt in long_train_receipts:
        total += len(receipt)
        count += 1
    print('Average long term receipt length:', total / count)
    print('Number receipts:', len(ret))
    total = .0
    count = 0
    for receipt in short_train_receipts:
        total += len(receipt)
        count += 1
    print('Average short term receipt length:', total / count)
    print('Number receipts:', len(ret))




    transaction_labelled_writer.close()
    transaction_unlabelled_writer.close()
    transaction_train_unlabelled_writer.close()
    transaction_train_labelled_writer.close()
    transaction_test_unlabelled_writer.close()
    transaction_test_labelled_writer.close()

    with open(os.path.join(folder_path, 'tr_long'), 'w') as f:
        for receipt in long_train_receipts:
            f.write(' '.join(receipt)+'\n')
    with open(os.path.join(folder_path, 'tr_short'), 'w') as f:
        for receipt in short_train_receipts:
            f.write(' '.join(receipt)+'\n')
    print('total long term train receipts:', len(long_train_receipts))
    print('total short term train receipts:', len(short_train_receipts))
    with open(os.path.join(folder_path, 'te_long'), 'w') as f:
        for receipt in test_long_receipts:
            f.write(' '.join(receipt)+'\n')
    with open(os.path.join(folder_path, 'te_short'), 'w') as f:
        for receipt in test_short_receipts:
            f.write(' '.join(receipt)+'\n')
    print('total long term test receipts:', len(test_long_receipts))
    print('total short term test receipts:', len(test_short_receipts))

def GenTrainAndTestSet(folder_path, file_name, ratio=0.9, verbose=False, is_first_col_index=False):
    ret = {}
    transaction_file_path = os.path.join(folder_path, file_name)
    print(transaction_file_path)
    with open(transaction_file_path, 'r') as f:
        for line in f:
            if is_first_col_index:
                user_id, item_id, quantity = line.strip().split(',')[1:]
            else:
                user_id, item_id, quantity = line.strip().split(',')
            item_id = Encode(item_id)
            if user_id in ret.keys():
                ret[user_id].append(item_id)
            else:
                ret[user_id] = [item_id]
    if verbose:
        total = .0
        count = 0
        for key in ret:
            total += len(ret[key])
            count += 1
        print('Average receipt length:', total/count)
        print('Number receipts:', len(ret))
    train_receipts = []
    test_receipts = []
    for user in ret.keys():
        if random.random() < ratio:
            train_receipts.append(ret[user])
        else:
            test_receipts.append(ret[user])
    with open(os.path.join(folder_path, 'tr_'+file_name), 'w') as f:
        for receipt in train_receipts:
            f.write(' '.join(receipt)+'\n')
    print('total train receipts:', len(train_receipts))
    with open(os.path.join(folder_path, 'te_'+file_name), 'w') as f:
        for receipt in test_receipts:
            f.write(' '.join(receipt)+'\n')
    print('total test receipts:', len(test_receipts))
    # write to disk


def Train(input_file_path, output_file_path, window=3, dim=70, cbow=True,
              sample=True, scalar=1, workers=8, threshold=300, use_gensim=True, hs=0, model_name=''):
    print('Training file:', input_file_path)
    Embedding(input_file_path, output_file_path, window=window, dim=dim, cbow=cbow,
              sample=sample, scalar=scalar, workers=workers, threshold=threshold, use_gensim=use_gensim, hs=hs, model_name=model_name)
    print()




if __name__ == "__main__":
    cur_dir = os.path.dirname(__file__)
    root = os.path.join(cur_dir)
    input_file_folder = os.path.join(root, 'online_shopping')
    input_file_path = os.path.join('cleaned')
    GenTrainAndTest(input_file_folder, input_file_path, ratio=0.9, verbose=True, resolution='daily', time_stamp_format='%m/%d/%Y %H:%M', ignore_first_row=True)