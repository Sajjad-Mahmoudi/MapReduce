with open('A.txt') as A, open('A_with_rowIndex.txt', 'w') as AIdx:
    for row_index, line in enumerate(A):
        line = line.rstrip()  # remove newline
        line = str(row_index) + ' ' + line  # append row index
        AIdx.write(line + '\n')  # write line