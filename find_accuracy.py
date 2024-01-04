import argparse


def read_lines_and_split(file_path):
    with open(file_path, 'r', encoding='utf-8') as fin:
        lines = [line.strip().split('\t') for line in fin.readlines() if line.strip()]
    return lines


def compare_gold_predicted(lines):
    correct = 0
    total = len(lines)
    correctList, incorrectList = list(), list()
    for indexLine, line in enumerate(lines):
        first_equation_items = set(line[0].split())
        second_equation_items = set(line[1].split())
        if line[0] == line[1]:
            correct += 1
            correctList.append(str(indexLine))
        else:
            incorrectList.append(str(indexLine))
        # if not first_equation_items - second_equation_items:
        #     correct += 1
        #     correctList.append(str(indexLine))
        # else:
        #     incorrectList.append(str(indexLine))
    return correct / total, correctList, incorrectList


def write_list_to_file(data_list, file_path):
    with open(file_path, 'w') as file_write:
        file_write.write('\n'.join(data_list))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred', dest='pred', help='Enter the predicted output')
    args = parser.parse_args()
    gold_predicted_data = read_lines_and_split(args.pred)
    returned_accuracy, correctList, incorrectList = compare_gold_predicted(gold_predicted_data)
    print(len(correctList), len(incorrectList))
    print('Test Accuracy={:.3f}'.format(returned_accuracy * 100))
