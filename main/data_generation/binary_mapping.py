import numpy as np
import json


binary_digits = 7
num_samples = 2 ** binary_digits  


binary_to_number_mapping = {}

for i in range(num_samples):
    binary = format(i, f'0{binary_digits}b')  
    number = np.random.uniform(0.0, 10.0)      
    binary_to_number_mapping[binary] = number


with open('JSON_mappings/7_digits.json', 'w') as json_file:
    json.dump(binary_to_number_mapping, json_file) #7_Digits is old, 7_digits is new (Jan 2024)

print(binary_to_number_mapping)

