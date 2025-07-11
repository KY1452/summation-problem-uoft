import json

# Load the JSON file
with open('JSON_mappings/7_digits.json', 'r') as file:
    data = json.load(file)

# Save the formatted data to a new JSON file
with open('JSON_mappings/7_digits.json', 'w') as file:
    json.dump(data, file, indent=4)
