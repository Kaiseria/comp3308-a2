import csv
import math

def classify_dt(training_filename, testing_filename):

    def read_csv(filename):
        data = []
        with open(filename, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                data.append([x.strip() for x in row])
        return data

    def majority_class(rows):
        died = 0
        survived = 0

        for row in rows:
            if row[-1] == "died":
                died += 1
            else:
                survived += 1

        if died >= survived:
            return "died"
        else:
            return "survived"

    def all_same_class(rows):
        first_class = rows[0][-1]

        for row in rows:
            if row[-1] != first_class:
                return False

        return True

    def entropy(rows):
        died = 0
        survived = 0

        for row in rows:
            if row[-1] == "died":
                died += 1
            else:
                survived += 1

        total = died + survived

        if died == 0 or survived == 0:
            return 0

        p_died = died / total
        p_survived = survived / total

        return -(
            p_died * math.log2(p_died)
            + p_survived * math.log2(p_survived)
        )

    def information_gain(rows, attribute_index):
        parent_entropy = entropy(rows)

        groups = {}

        for row in rows:
            value = row[attribute_index]

            if value not in groups:
                groups[value] = []

            groups[value].append(row)

        weighted_entropy = 0
        total = len(rows)

        for group in groups.values():
            weighted_entropy += len(group) / total * entropy(group)

        return parent_entropy - weighted_entropy

    def best_attribute(rows, attributes):
        best_attr = attributes[0]
        best_gain = information_gain(rows, best_attr)

        for attr in attributes[1:]:
            gain = information_gain(rows, attr)

            if gain > best_gain:
                best_gain = gain
                best_attr = attr

        return best_attr

    def build_tree(rows, attributes, default_class):

        if len(rows) == 0:
            return default_class

        if all_same_class(rows):
            return rows[0][-1]

        if len(attributes) == 0:
            return majority_class(rows)

        current_majority = majority_class(rows)

        attr = best_attribute(rows, attributes)

        tree = {
            "attribute": attr,
            "majority": current_majority,
            "children": {}
        }

        values = []

        for row in rows:
            if row[attr] not in values:
                values.append(row[attr])

        remaining_attributes = []

        for a in attributes:
            if a != attr:
                remaining_attributes.append(a)

        for value in values:
            subset = []

            for row in rows:
                if row[attr] == value:
                    subset.append(row)

            tree["children"][value] = build_tree(
                subset,
                remaining_attributes,
                current_majority
            )

        return tree

    def predict(tree, row):

        if tree == "died" or tree == "survived":
            return tree

        attr = tree["attribute"]
        value = row[attr]

        if value not in tree["children"]:
            return tree["majority"]

        return predict(tree["children"][value], row)

    training_data = read_csv(training_filename)
    testing_data = read_csv(testing_filename)

    num_features = len(training_data[0]) - 1

    attributes = []

    for i in range(num_features):
        attributes.append(i)

    tree = build_tree(training_data, attributes, majority_class(training_data))

    predictions = []

    for row in testing_data:
        predictions.append(predict(tree, row))

    return predictions

def read_folds(folds_filename):
    folds = []
    current_fold = []

    with open(folds_filename, "r") as f:
        reader = csv.reader(f)

        for row in reader:
            if len(row) == 0:
                continue

            if row[0].startswith("fold"):
                if current_fold != []:
                    folds.append(current_fold)
                current_fold = []
            else:
                current_fold.append(row)

        if current_fold != []:
            folds.append(current_fold)

    return folds


def write_csv(filename, rows):
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)


def evaluate_dt_10fold(folds_filename):
    folds = read_folds(folds_filename)

    accuracies = []

    for i in range(10):
        testing_with_class = folds[i]

        training = []

        for j in range(10):
            if j != i:
                training.extend(folds[j])

        testing_without_class = []

        for row in testing_with_class:
            testing_without_class.append(row[:-1])

        write_csv("train_temp.csv", training)
        write_csv("test_temp.csv", testing_without_class)

        predictions = classify_dt("train_temp.csv", "test_temp.csv")

        correct = 0

        for prediction, actual_row in zip(predictions, testing_with_class):
            actual_class = actual_row[-1]

            if prediction == actual_class:
                correct += 1

        accuracy = correct / len(testing_with_class)
        accuracies.append(accuracy)

        print("Fold", i + 1, "accuracy:", accuracy)

    average_accuracy = sum(accuracies) / len(accuracies)

    print("Average accuracy:", average_accuracy)

    return average_accuracy

evaluate_dt_10fold("heart-folds.csv")