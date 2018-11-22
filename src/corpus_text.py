from paths import *


def read_data(file_path):
    data = []
    with open(file_path) as f:
        for line in f:
            line = line.strip('\n').split('\t')
            data.append(line[2])
    return data


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


if __name__ == "__main__":
    main()
