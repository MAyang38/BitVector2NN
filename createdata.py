import csv
import pandas as pd
import numpy as np


def bitwise_addition(x, y):
    return x + y


def bitwise_subtraction(x, y):
    return x - y


def bitwise_multiplication(x, y):
    return x * y


def bitwise_division(x, y):
    return x // y


def bitwise_and(x, y):
    return x & y


def bitwise_or(x, y):
    return x | y


def bitwise_not(x):
    return ~x


def perform_bitwise_operations(x, y):
    operations = {
        'Addition': bitwise_addition,
        'Subtraction': bitwise_subtraction,
        'Multiplication': bitwise_multiplication,
        'Division': bitwise_division,
        'AND': bitwise_and,
        'OR': bitwise_or,
        'NOT': bitwise_not
    }

    # Create one-hot encoding for operations
    operation_labels = list(operations.keys())
    operation_one_hot = np.eye(len(operation_labels))

    # Convert operation labels to one-hot vectors
    operation_vectors = {}
    for i, label in enumerate(operation_labels):
        operation_vectors[label] = operation_one_hot[i]

    # Perform bitwise operations and store in a dataframe
    rows = []
    for operator, operation_func in operations.items():
        if operator == 'NOT':
            result = operation_func(x)
            rows.append([format(x, '064b'), format(0, '064b'), operation_vectors[operator], format(result, '064b')])
        else:
            result = operation_func(x, y)
            rows.append([format(x, '064b'), format(y, '064b'), operation_vectors[operator], format(result, '064b')])

    # Create a dataframe from the rows
    df = pd.DataFrame(rows, columns=['Operand 1', 'Operand 2', 'Operator', 'Result'])

    # Save the dataframe to a CSV file
    df.to_csv('bitwise_operations.csv', index=False)


# Example usage
operand1 = 0b10101010  # Replace with your desired operand 1
operand2 = 0b11001100  # Replace with your desired operand 2

perform_bitwise_operations(operand1, operand2)
