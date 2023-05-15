import csv
import json
import unittest

""" 16.) Make .json file from .csv"""
# Open the CSV file
with open('podatki_python_akademijaCSV.csv', newline='') as csvfile:
    # Read the first line of the CSV file and extract the field names
    fieldnames = csvfile.readline().strip().split(';')
    # Create a list to store the data
    data_list = []
    # Read the remaining lines of the CSV file and create a list of values for each line
    reader = csv.reader(csvfile, delimiter=';', quoting=csv.QUOTE_ALL)
    for row in reader:
        # Create a dictionary from the row values and append it to the data list
        row_dict = dict(zip(fieldnames, row))
        data_list.append(row_dict)

# Write the data list to a JSON file
with open('one_line_data.json', 'w') as jsonfile:
    json.dump(data_list, jsonfile)


""" 17.) Make .csv file from .json"""
# Load the JSON file
with open('one_line_data.json') as jsonfile:
    data = json.load(jsonfile)

# Open the CSV file for writing
with open('one_line_data.csv', 'w', newline='') as csvfile:
    # Create a writer object
    writer = csv.writer(csvfile, delimiter=';')
    # Write the column headers to the CSV file
    writer.writerow(data[0].keys())
    # Write each row of data to the CSV file
    for row in data:
        writer.writerow(row.values())


""" 19.) Unit test"""
class TestCSVConversion(unittest.TestCase):
    def test_csv_conversion(self):
        # Read the contents of the original CSV file
        with open('podatki_python_akademijaCSV.csv', 'r') as f:
            original_csv = f.read()

        # Read the contents of the new CSV file
        with open('one_line_data.csv', 'r') as f:
            new_csv = f.read()

        # Compare the contents of the two CSV files
        self.assertEqual(original_csv, new_csv)

if __name__ == '__main__':
    unittest.main()
