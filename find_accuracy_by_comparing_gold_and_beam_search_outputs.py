"""Find accuracy by comparing gold equations with outputs predicted using beam search."""
import argparse


def read_lines_from_file(file_path):
    """Read lines from a file."""
    with open(file_path, 'r', encoding='utf-8') as fin:
        return [line.strip() for line in fin.readlines() if line.strip()]


def compare_gold_and_predicted(gold_equations, pred_outputs):
    """Compare gold and predicted equations."""
    correct = 0
    correct_best = 0
    total = len(gold_equations)
    for index, gold_equation in enumerate(gold_equations):
        pred_output = pred_outputs[index]
        pred_output = pred_output.replace(' END', '')
        pred_equations = pred_output.split('\t')
        if gold_equation == pred_equations[0]:
            correct_best += 1
        if gold_equation in pred_equations:
            correct += 1
    print('Total Correct Equations in Beam 3 =', correct)
    print('Total Correct Equations in Beam 1 =', correct_best)
    print('Total Equations =', total)
    return correct / total, correct_best / total


def write_list_to_file(data_list, file_path):
    with open(file_path, 'w') as file_write:
        file_write.write('\n'.join(data_list))


def main():
    """Pass arguments and call functions here."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--gold', dest='gold', help='Enter the gold output')
    parser.add_argument('--pred', dest='pred', help='Enter the predicted output')
    args = parser.parse_args()
    gold_equations = read_lines_from_file(args.gold)
    predicted_data = read_lines_from_file(args.pred)
    returned_accuracy = compare_gold_and_predicted(gold_equations, predicted_data)
    print('Test Accuracy with Beam size 3 = {:.3f}'.format(returned_accuracy[0] * 100))
    print('Test Accuracy with Beam size 1 = {:.3f}'.format(returned_accuracy[1] * 100))


if __name__ == '__main__':
    main()
