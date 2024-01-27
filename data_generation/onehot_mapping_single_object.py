import json
import random

def generate_one_hot_vector(index, length=128):
    return [1 if i == index else 0 for i in range(length)]

def generate_random_mapping():
    mappings = {}
    for i in range(128):
        one_hot_vector = generate_one_hot_vector(i)
        # Convert the one-hot vector to a string format (as list cannot be a JSON key)
        key = ''.join(map(str, one_hot_vector))
        value = round(random.uniform(0, 20), 5)  # Rounded to 5 decimal places
        mappings[key] = value
    return mappings

if __name__ == "__main__":
    mapping = generate_random_mapping()
    with open("128_digits_20.json", "w") as f:
        json.dump(mapping, f, indent=4)

    print("Mapping saved to mapping.json!")
