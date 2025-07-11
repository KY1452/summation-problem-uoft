# import numpy as np
# import itertools
# import random
# import json
# import pandas as pd

# # Set seeds for reproducibility
# np.random.seed(42)
# random.seed(42)

# # Load the binary to number mapping
# with open('JSON_mappings/128_digits.json', 'r') as json_file:
#     binary_to_number_mapping = json.load(json_file)

# binary_digits = 128
# binary_combinations = [format(2**i, f'0{binary_digits}b') for i in range(binary_digits)]

# def efficient_random_combination_sample(iterable, repeat, sample_size):
#     sampled_combinations = set()  # Use a set to store unique combinations
#     while len(sampled_combinations) < sample_size:
#         combination = random.sample(iterable, k=repeat)
#         combination_tuple = tuple(combination)
#         if combination_tuple not in sampled_combinations:
#             sampled_combinations.add(combination_tuple)
#     return [list(combination) for combination in sampled_combinations]

# input_combinations_sample = efficient_random_combination_sample(binary_combinations, 5, 1000)

# # input_data = np.array([[int(bit) for bit in combination[0] + combination[1] + combination[2] + combination[3] + combination[4]] for combination in input_combinations_sample])
# # input_numbers = np.array([binary_to_number_mapping[combination[0]] + binary_to_number_mapping[combination[1]] + binary_to_number_mapping[combination[2]] + binary_to_number_mapping[combination[3]] + binary_to_number_mapping[combination[4]] for combination in input_combinations_sample])
# input_data = np.array([[int(bit) for bit in combination[0] + combination[1] + combination[2] + combination[3] + combination[4]] for combination in input_combinations_sample])
# input_numbers = np.array([[binary_to_number_mapping[combination[0]] + binary_to_number_mapping[combination[1]] + binary_to_number_mapping[combination[2]] + binary_to_number_mapping[combination[3]] + binary_to_number_mapping[combination[4]]] for combination in input_combinations_sample])
# output_data = input_numbers.reshape((-1, 1))
# print(input_data[0])
# np.save('input_data_6.npy', input_data)
# np.save('output_data_6.npy', output_data)


# import numpy as np
# import itertools
# import random
# import json

# # Set seeds for reproducibility
# np.random.seed(42)
# random.seed(42)

# # Load the binary to number mapping
# with open('JSON_mappings/7_digits.json', 'r') as json_file:
#     binary_to_number_mapping = json.load(json_file)

# # Generate all 128 unique 7-bit binary combinations
# binary_combinations = list(binary_to_number_mapping.keys())

# # Function to sample combinations
# def efficient_random_combination_sample(iterable, repeat, sample_size):
#     sampled_combinations = set()
#     while len(sampled_combinations) < sample_size:
#         combination = random.sample(iterable, k=repeat)
#         combination_tuple = tuple(combination)
#         if combination_tuple not in sampled_combinations:
#             sampled_combinations.add(combination_tuple)
#     return [list(combination) for combination in sampled_combinations]

# # Sample 1000 pairs
# input_combinations_sample = efficient_random_combination_sample(binary_combinations, 10, 1000)

# # Prepare input and output data
# input_data = np.array([[int(bit) for bit in combination[0] + combination[1] + combination[2] + combination[3] + combination[4] + combination[5] + combination[6] + combination[7] + combination[8] + combination[9]] for combination in input_combinations_sample])
# input_numbers = np.array([[binary_to_number_mapping[combination[0]] + binary_to_number_mapping[combination[1]] + binary_to_number_mapping[combination[2]] + binary_to_number_mapping[combination[3]] + binary_to_number_mapping[combination[4]] + binary_to_number_mapping[combination[5]] + binary_to_number_mapping[combination[6]] + binary_to_number_mapping[combination[7]] + binary_to_number_mapping[combination[8]] + binary_to_number_mapping[combination[9]]] for combination in input_combinations_sample])
# output_data = input_numbers.reshape((-1, 1))

# # Save the data
# np.save('numpy_arrays/input_data_5.npy', input_data)
# np.save('numpy_arrays/output_data_5.npy', output_data)

# # Print the format of one output and input data point
# print("Input Data Sample:", input_data[0])
# print("Output Data Sample:", output_data[0])

import numpy as np
import random
import json

# Set seeds for reproducibility
np.random.seed(42)
random.seed(42)

# Load the binary to number mapping
with open('JSON_mappings/7_digits.json', 'r') as json_file:
    binary_to_number_mapping = json.load(json_file)

# Generate all 128 unique 7-bit binary combinations
binary_combinations = list(binary_to_number_mapping.keys())

# Function to sample combinations
def efficient_random_combination_sample(iterable, repeat, sample_size):
    sampled_combinations = set()
    while len(sampled_combinations) < sample_size:
        combination = random.sample(iterable, k=repeat)
        combination_tuple = tuple(combination)
        if combination_tuple not in sampled_combinations:
            sampled_combinations.add(combination_tuple)
    return [list(combination) for combination in sampled_combinations]

# Sample 30,000 sets with each set containing 60 numbers
input_combinations_sample = efficient_random_combination_sample(binary_combinations, 60, 30000)

# Prepare input and output data
input_data = np.array([[int(bit) for bit in ''.join(combination)] for combination in input_combinations_sample])
input_numbers = np.array([[sum(binary_to_number_mapping[num] for num in combination)] for combination in input_combinations_sample])
output_data = input_numbers.reshape((-1, 1))

# Save the data
np.save('numpy_arrays/input_data_60.npy', input_data)
np.save('numpy_arrays/output_data_60.npy', output_data)

# Print the format of one output and input data point
print("Input Data Sample:", input_data[0])
print("Output Data Sample:", output_data[0])

