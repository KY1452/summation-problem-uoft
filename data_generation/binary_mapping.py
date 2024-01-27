import numpy as np
import json


binary_digits = 7
num_samples = 2 ** binary_digits  


binary_to_number_mapping = {}

for i in range(num_samples):
    binary = format(i, f'0{binary_digits}b')  
    number = np.random.uniform(0.0, 20.0)      
    binary_to_number_mapping[binary] = number


with open('128_digits_20.json', 'w') as json_file:
    json.dump(binary_to_number_mapping, json_file)


print(binary_to_number_mapping)

