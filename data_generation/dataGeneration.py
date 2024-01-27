import numpy as np
import itertools
import random
import json
import pandas as pd

# Set seeds for reproducibility
np.random.seed(42)
random.seed(42)

# Load the binary to number mapping
with open('128_digits.json', 'r') as json_file:
    binary_to_number_mapping = json.load(json_file)

binary_digits = 128
binary_combinations = [format(2**i, f'0{binary_digits}b') for i in range(binary_digits)]

def efficient_random_combination_sample(iterable, repeat, sample_size):
    sampled_combinations = set()  # Use a set to store unique combinations
    while len(sampled_combinations) < sample_size:
        combination = random.sample(iterable, k=repeat)
        combination_tuple = tuple(combination)
        if combination_tuple not in sampled_combinations:
            sampled_combinations.add(combination_tuple)
    return [list(combination) for combination in sampled_combinations]

input_combinations_sample = efficient_random_combination_sample(binary_combinations, 2, 1000)

# input_data = np.array([[int(bit) for bit in combination[0] + combination[1] + combination[2] + combination[3] + combination[4]] for combination in input_combinations_sample])
# input_numbers = np.array([binary_to_number_mapping[combination[0]] + binary_to_number_mapping[combination[1]] + binary_to_number_mapping[combination[2]] + binary_to_number_mapping[combination[3]] + binary_to_number_mapping[combination[4]] for combination in input_combinations_sample])
input_data = np.array([[int(bit) for bit in combination[0] + combination[1]] for combination in input_combinations_sample])
input_numbers = np.array([[binary_to_number_mapping[combination[0]] + binary_to_number_mapping[combination[1]]] for combination in input_combinations_sample])
output_data = input_numbers.reshape((-1, 1))

np.save('input_data_3.npy', input_data)
np.save('output_data_3.npy', output_data)

