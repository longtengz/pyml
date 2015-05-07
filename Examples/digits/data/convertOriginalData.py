import os

def old2new(path, writeFileObj):
    fileList = os.listdir(path)

    for each_file in fileList:
        which_number = each_file[0]
        with open(path + '/' + each_file, 'r') as file_obj:
            writeFileObj.write(which_number + '-')
            for each_line in file_obj:
                writeFileObj.write(each_line[:-1])
            writeFileObj.write('\n')


with open('./testDigits.data', 'w') as testDigitsFile:
    old2new('./originalData/testDigits', testDigitsFile)

with open('./trainingDigits.data', 'w') as trainingDigitsFile:
    old2new('./originalData/trainingDigits', trainingDigitsFile)
