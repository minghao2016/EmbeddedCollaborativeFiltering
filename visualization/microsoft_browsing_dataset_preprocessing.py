"""
Rec-sys Embedding Rec-sys prototyping

__author__:
    charles@qileap

"""
def convert_to_csv(file_path, output_path):
    transactions = {}
    current_user = ''
    out_stream = open(output_path, 'w')
    with open(file_path) as f:
        for line in f:
            data = line.split(',')
            if data[0] == 'C':
                current_user = data[1].replace('\"', '')
                transactions[current_user] = []
            elif data[0] == 'V':
                item_id = data[1]
                transactions[current_user].append(item_id)
                out_stream.write(','.join([current_user, item_id, '1\n']))
    out_stream.close()
    return transactions

if __name__ == "__main__":
    output_path = 'data/microsoft_browsing.csv'
    input_path = 'data/microsoft_browsing_data.dat'
    convert_to_csv(input_path, output_path)