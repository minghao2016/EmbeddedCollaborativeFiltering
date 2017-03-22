import csv

def get_columns_movie_lens(csv_path, output_path, cols, delimiter=',', quotechar='"'):
    output_stream = open(output_path, 'w')
    with open(csv_path, 'r') as f:
        reader = csv.reader(f, delimiter=delimiter, quotechar=quotechar)
        writer = csv.writer(output_stream, delimiter=',', quotechar=quotechar)
        for rows in reader:
            if int(rows[2]) >= 4:
                row = [rows[i] for i in cols]
                row.insert(2, 1)
            else:
                continue
            writer.writerow(row)
    output_stream.close()


def get_columns(csv_paths, output_path, cols, delimiter=',', quotechar='"'):
    output_stream = open(output_path, 'w')
    for csv_path in csv_paths:
        with open(csv_path, 'r') as f:
            reader = csv.reader(f, delimiter=delimiter, quotechar=quotechar)
            writer = csv.writer(output_stream, delimiter=',', quotechar=quotechar)
            for rows in reader:
                row = [rows[i].strip() for i in cols]
                writer.writerow(row)
    output_stream.close()





if __name__ == "__main__":
    import os
    project = 'movie_lens_100k'
    project = 'tafeng'
    project = 'gift_receipt'
    cur_dir = os.path.dirname(__file__)
    # gift_store_indices
    input_path = [os.path.join(cur_dir, project, 'raw')]
    output_path = os.path.join(cur_dir, project, 'cleaned')
    columns = [0, 2, 4, 5]
    get_columns(input_path, output_path, columns)



    # movie lens
    # input_path = [os.path.join(cur_dir, project, 'raw')]
    # output_path = os.path.join(cur_dir, project, 'cleaned')
    # columns = [0, 1, 3]
    # get_columns_movie_lens([input_path], output_path, columns, delimiter='\t')


    # tafeng
    # input_path = [os.path.join(cur_dir, project, 'D'+x) for x in ['01', '02', '11', '12']]
    # output_path = os.path.join(cur_dir, project, 'cleaned')
    # columns = [1, 5, 6, 0]
    # get_columns(input_path, output_path, columns, delimiter=';')


    # gift_store_indices
    # input_path = [os.path.join(cur_dir, project, 'raw')]
    # output_path = os.path.join(cur_dir, project, 'cleaned')
    # columns = [6, 2, 4, 5]
    # get_columns(input_path, output_path, columns)
