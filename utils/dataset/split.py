import random

def split_dataset(input_file, train_file, test_file, validation_file, train_ratio=0.8, test_ratio=0.1, validation_ratio=0.1):
    with open(input_file, 'r', encoding='utf-8') as infile:
        lines = infile.readlines()
    
    # Shuffle lines to ensure random distribution
    random.shuffle(lines)
    
    total_lines = len(lines)
    train_end = int(total_lines * train_ratio)
    test_end = train_end + int(total_lines * test_ratio)
    
    # Splitting the dataset
    train_lines = lines[:train_end]
    test_lines = lines[train_end:test_end]
    validation_lines = lines[test_end:]
    
    # Writing the splits to their respective files
    with open(train_file, 'w', encoding='utf-8') as outfile:
        outfile.writelines(train_lines)
    with open(test_file, 'w', encoding='utf-8') as outfile:
        outfile.writelines(test_lines)
    with open(validation_file, 'w', encoding='utf-8') as outfile:
        outfile.writelines(validation_lines)

# Usage example
input_file = 'filelists_phoneme.txt'  # The path to your corrected input file
train_output_file = 'train_set.txt'
test_output_file = 'test_set.txt'
validation_output_file = 'validation_set.txt'

split_dataset(input_file, train_output_file, test_output_file, validation_output_file)
