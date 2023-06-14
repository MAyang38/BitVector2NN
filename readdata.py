import csv
import numpy as np


def convert_binary_to_vector(binary_string):
    return [int(bit) for bit in binary_string]

def convert_operator_to_vector(operator_string):
    operator_array = np.fromstring(operator_string[1:-1], dtype=np.float32, sep=' ')
    return operator_array
# def convert_operator_to_vector(operator):
#     operators = ['Addition', 'Subtraction', 'Multiplication', 'Division', 'AND', 'OR', 'NOT']
#     operator_vector = [0] * len(operators)
#     operator_index = operators.index(operator)
#     operator_vector[operator_index] = 1
#     return operator_vector


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
            # operator = [int(bit) for bit in row[2]]
            # operator = [int(bit) for bit in row[2]]
            operator = convert_operator_to_vector(row[2])
            # operator = np.array(row[2], dtype=np.float32)
            # operator = convert_operator_to_vector(row[2])
            result = convert_binary_to_vector(row[3])

            # Concatenate operand1, operand2, and operator to form input vector x
            # x.append(operand1 + operand2 + operator)
            x.append(operand1 + operand2 + list(operator) + result)
            # Append result as output vector y
            y.append(result)

    return np.array(x), np.array(y)


# Example usage
csv_file = 'bitwise_operations.csv'  # Replace with your CSV file path

x_data, y_data = read_csv_data(csv_file)
print(x_data)
print(y_data)
print("Input shape:", x_data.shape)
print("Output shape:", y_data.shape)
