# import json
# import random

# def generate_one_hot_vector(index, length=128):
#     return [1 if i == index else 0 for i in range(length)]

# def generate_random_mapping():
#     mappings = {}
#     for i in range(128):
#         one_hot_vector = generate_one_hot_vector(i)
#         # Convert the one-hot vector to a string format (as list cannot be a JSON key)
#         key = ''.join(map(str, one_hot_vector))
#         value = round(random.uniform(0, 10), 5)  # Rounded to 5 decimal places
#         mappings[key] = value
#     return mappings

# if __name__ == "__main__":
#     mapping = generate_random_mapping()
#     with open("128_digits_20.json", "w") as f:
#         json.dump(mapping, f, indent=4)

#     print("Mapping saved to mapping.json!")


import json
import random

def generate_one_hot_vector(index, length=128):
    # Special case for the first entry to be all zeros
    if index == -1:
        return [0 for _ in range(length)]
    return [1 if i == index else 0 for i in range(length)]

def generate_random_mapping():
    mappings = {}
    # Start with the special case of all zeros
    one_hot_vector = generate_one_hot_vector(-1)
    key = ''.join(map(str, one_hot_vector))
    value = round(random.uniform(0, 10), 5)  # Rounded to 5 decimal places
    mappings[key] = value
    # Continue with the one-hot encoding for the rest
    for i in range(128):
        one_hot_vector = generate_one_hot_vector(i)
        # Convert the one-hot vector to a string format (as list cannot be a JSON key)
        key = ''.join(map(str, one_hot_vector))
        value = round(random.uniform(0, 10), 5)  # Rounded to 5 decimal places
        mappings[key] = value
    return mappings

if __name__ == "__main__":
    mapping = generate_random_mapping()
    with open("128_digits_20.json", "w") as f:
        json.dump(mapping, f, indent=4)

    print("Mapping saved to 128_digits_20.json!")

