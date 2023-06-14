import csv
import numpy as np


def convert_binary_to_vector(binary_string):
    return np.array([int(bit) for bit in binary_string], dtype=np.int8)


def convert_operator_to_vector(operator_string):
    operator_array = np.fromstring(operator_string[1:-1], dtype=np.float32, sep=' ')
    return operator_array


def read_csv_data(file_path):
    x = []
    y = []

    with open(file_path, 'r') as csvfile:
        reader = csv.reader(csvfile)

        # Skip header
        next(reader)

        for row in reader:
            operand1 = convert_binary_to_vector(row[0])
            operand2 = convert_binary_to_vector(row[1])
            operator = convert_operator_to_vector(row[2])
            result = convert_binary_to_vector(row[3])

            # Flatten operand1, operand2, operator, and result into a single vector x
            x.append(np.concatenate([operand1, operand2, operator, result]))
            y.append(result)
    return np.array(x),np.array(y)


# Example usage
csv_file = 'bitwise_operations.csv'  # Replace with your CSV file path

x_data, y_data= read_csv_data(csv_file)
print(x_data)
print(y_data)
print("Input shape:", x_data.shape)
print("output shape:", y_data.shape)
