from paths import *


def read_data(file_path):
    data = []
    with open(file_path) as f:
        for line in f:
            line = line.strip('\n').split('\t')
            data.append(line[2])
    return data


def split_data():
    train_data = read_data(TRAIN_PATH)
    test_data = read_data(TEST_PATH)
    valid_data = read_data(VALID_PATH)
    
    for i, line in enumerate(train_data):
        with open("../data/split/train/text%d.txt" % i, 'w') as f:
            f.write(line)

    for i, line in enumerate(test_data):
        with open("../data/split/test/text%d.txt" % i, 'w') as f:
            f.write(line)

    for i, line in enumerate(valid_data):
        with open("../data/split/valid/text%d.txt" % i, 'w') as f:
            f.write(line)

def write_data(data, file_path):
    with open(file_path, 'w') as f:
        for line in data:
            f.write(line+"\n")


def main():

    train_data = read_data(TRAIN_PATH)
    test_data = read_data(TEST_PATH)
    valid_data = read_data(VALID_PATH)

    write_data(train_data, TRAIN_RES_PATH)
    write_data(test_data, TEST_RES_PATH)
    write_data(valid_data, VALID_RES_PATH)

    split_data()


if __name__ == "__main__":
    main()
