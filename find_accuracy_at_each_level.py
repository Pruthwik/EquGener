import argparse


def read_lines_and_split(file_path):
    with open(file_path, 'r', encoding='utf-8') as fin:
        lines = [line.strip().split('\t') for line in fin.readlines() if line.strip()]
    return lines


def compare_gold_predicted(lines):
    correct_whole = 0
    correct_components = 0
    correct_operand1 = 0
    correct_operand2 = 0
    correct_operator = 0
    total = len(lines)
    correctList, incorrectList = list(), list()
    for indexLine, line in enumerate(lines):
        first_equation_items = set(line[0].split())
        first_equation_operand1 = line[0].split()[0]
        first_equation_operand2 = line[0].split()[1]
        first_equation_operator = line[0].split()[2]
        second_equation_items = set(line[1].split())
        second_equation_operand1 = line[1].split()[0]
        second_equation_operand2 = line[1].split()[1]
        second_equation_operator = line[1].split()[2]
        if not first_equation_items - second_equation_items:
            correct_components += 1
        if line[0] == line[1]:
            correct_whole += 1
            correctList.append(str(indexLine))
        else:
            incorrectList.append(str(indexLine))
        if first_equation_operand1 == second_equation_operand1:
            correct_operand1 += 1
        if first_equation_operand2 == second_equation_operand2:
            correct_operand2 += 1
        if first_equation_operator == second_equation_operator:
            correct_operator += 1
    return correct_whole / total, correct_components / total, correct_operand1 / total, correct_operand2 / total, correct_operator / total, correctList, incorrectList


def write_list_to_file(data_list, file_path):
    with open(file_path, 'w') as file_write:
        file_write.write('\n'.join(data_list))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred', dest='pred', help='Enter the predicted output')
    args = parser.parse_args()
    gold_predicted_data = read_lines_and_split(args.pred)
    returned_accuracy, returned_components_accuracy, returned_op1_accuracy, returned_op2_accuracy, returned_opr_accuracy, correctList, incorrectList = compare_gold_predicted(gold_predicted_data)
    print('Whole Equation Test Accuracy={:.3f}'.format(returned_accuracy * 100))
    print('All components in Equation Test Accuracy={:.3f}'.format(returned_components_accuracy * 100))
    print('Operand1 Test Accuracy={:.3f}'.format(returned_op1_accuracy * 100))
    print('Operand2 Test Accuracy={:.3f}'.format(returned_op2_accuracy * 100))
    print('Operator Test Accuracy={:.3f}'.format(returned_opr_accuracy * 100))
