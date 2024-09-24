''' This script counts the number of occurrences of the labels '0' and '1' in a file.
    indicating the number of lines that correspond to each speaker on each file.
'''
def count_occurrences(file_path):
    count_0 = 0
    count_1 = 0

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if '|0|' in line:
                count_0 += 1
            if '|1|' in line:
                count_1 += 1

    return count_0, count_1

# Replace 'file_path' with the path to your file
file_path_1 = 'test_set_ms.txt'
file_path_2 = 'train_set_ms.txt'
file_path_3 = 'validation_set_ms.txt'

counts_file_1 = count_occurrences(file_path_1)
counts_file_2 = count_occurrences(file_path_2)
counts_file_3 = count_occurrences(file_path_3)

print("Counts for file 1:", counts_file_1)
print("Counts for file 2:", counts_file_2)
print("Counts for file 3:", counts_file_3)