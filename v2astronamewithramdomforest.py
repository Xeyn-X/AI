import pandas as pd

# Function to read business names from a CSV file
def read_business_names(file_path):
    name_list = []
    with open(file_path, 'r') as file:
        for i in file:
            cleaned_line = i.replace('\n', '')
            name_list.append(cleaned_line)
    return pd.DataFrame({'Name': name_list})

# Function to find the first and last consonants in a name
def find_consonants(name, consonant_to_day):
    name_characters = list(name)
    consonants = [ch for ch in name_characters if ch in consonant_to_day]
    first_consonant = consonant_to_day[consonants[0]] if len(consonants) > 0 else None

    if name_characters[-1] == '်':
        last_consonant = consonant_to_day[consonants[-2]] if len(consonants) > 0 else None
    elif name_characters[-2] == '်':
        last_consonant = consonant_to_day[consonants[-2]] if len(consonants) > 0 else None
    else:
        last_consonant = consonant_to_day[consonants[-1]] if len(consonants) > 0 else None

    return first_consonant, last_consonant
