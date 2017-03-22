import xlrd
import csv

def csv_from_excel(xls, tab_name, output_path):

    wb = xlrd.open_workbook(xls)
    sh = wb.sheet_by_name(tab_name)
    your_csv_file = open(output_path, 'density')
    wr = csv.writer(your_csv_file, quoting=csv.QUOTE_ALL)

    for rownum in range(sh.nrows):
        wr.writerow(sh.row_values(rownum))

    your_csv_file.close()

if __name__ == "__main__":
    input_file= 'data/Online Retail.xlsx'
    output_file = 'data/online_retails.csv'
    tab_name = 'Online Retail'
    csv_from_excel(input_file, tab_name, output_file)