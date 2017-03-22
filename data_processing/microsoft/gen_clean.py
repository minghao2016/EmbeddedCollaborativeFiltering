import csv, os

def gen_data_from_raw(input_path, output_path):
	output_stream = open(output_path, 'w')
	writer = csv.writer(output_stream, delimiter=',', quotechar='"')
	with open(input_path, 'r') as f:
		reader = csv.reader(f, delimiter=',', quotechar='"')
		user_id = None
		for row in reader:
			if row[0] == 'C':
				user_id = row[1]
			elif row[0] == 'V':
				item_id = row[1]
				writer.writerow([user_id, item_id, 1, 0])
	output_stream.close()
	pass


if __name__ == "__main__":
	cur_dir = os.path.dirname(__file__)
	file_name = os.path.join(cur_dir, 'anonymous-msweb.data')
	out_file = os.path.join(cur_dir, 'cleaned')
	gen_data_from_raw(file_name, out_file)