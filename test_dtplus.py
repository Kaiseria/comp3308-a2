import csv
import math

# MyDT+ is the modified decision tree. It uses gain ratio for splitting and
# pre-pruning with a maximum depth to reduce overfitting on the small dataset.

ATTRIBUTE_NAMES = [
    "age",
    "anaemia",
    "CPK",
    "diabetes",
    "ejection_fraction",
    "high_blood_pressure",
    "platelets",
    "serum_creatinine",
    "serum_sodium",
    "sex",
    "smoking"
]

CLASS_LABELS = ["died", "survived"]


def read_csv(filename):
    """Read a CSV file and trim whitespace from every cell."""
    data = []
    with open(filename, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            data.append([x.strip() for x in row])
    return data


def read_folds(folds_filename):
    """Read the provided stratified folds from heart-folds.csv."""
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
                current_fold.append([x.strip() for x in row])

        if current_fold != []:
            folds.append(current_fold)

    return folds


def class_counts(rows):
    """Count how many training rows belong to each class."""
    counts = {}

    for label in CLASS_LABELS:
        counts[label] = 0

    for row in rows:
        counts[row[-1]] += 1

    return counts


def majority_class(rows, default_class="died"):
    """Return the majority class, using default_class for empty subsets."""
    if len(rows) == 0:
        return default_class

    counts = class_counts(rows)

    if counts["died"] >= counts["survived"]:
        return "died"
    else:
        return "survived"


def all_same_class(rows):
    """Check whether all rows in a subset have the same class label."""
    first_class = rows[0][-1]

    for row in rows:
        if row[-1] != first_class:
            return False

    return True


def entropy(rows):
    """Calculate class entropy for a subset of rows."""
    counts = class_counts(rows)
    total = len(rows)
    result = 0

    for label in CLASS_LABELS:
        if counts[label] > 0:
            p = counts[label] / total
            result -= p * math.log2(p)

    return result


def group_by_attribute(rows, attribute_index):
    """Group rows by one attribute value before calculating a split score."""
    groups = {}

    for row in rows:
        value = row[attribute_index]

        if value not in groups:
            groups[value] = []

        groups[value].append(row)

    return groups


def information_gain(rows, attribute_index):
    """Calculate information gain for one candidate splitting attribute."""
    parent_entropy = entropy(rows)
    groups = group_by_attribute(rows, attribute_index)
    weighted_entropy = 0
    total = len(rows)

    for group in groups.values():
        weighted_entropy += len(group) / total * entropy(group)

    return parent_entropy - weighted_entropy


def split_information(rows, attribute_index):
    """Calculate split information, the normalising term in gain ratio."""
    groups = group_by_attribute(rows, attribute_index)
    total = len(rows)
    split_info = 0

    for group in groups.values():
        p = len(group) / total

        if p > 0:
            split_info -= p * math.log2(p)

    return split_info


def gain_ratio(rows, attribute_index):
    """Calculate gain ratio to reduce bias toward many-valued attributes."""
    split_info = split_information(rows, attribute_index)

    if split_info == 0:
        return 0

    return information_gain(rows, attribute_index) / split_info


def best_attribute(rows, attributes):
    """Choose the attribute with the highest gain ratio."""
    best_attr = attributes[0]
    best_score = gain_ratio(rows, best_attr)

    for attr in attributes[1:]:
        score = gain_ratio(rows, attr)

        if score > best_score:
            best_score = score
            best_attr = attr

    return best_attr


def make_leaf(rows, default_class):
    """Create a leaf node and store Weka-style count/error information."""
    prediction = majority_class(rows, default_class)
    counts = class_counts(rows)
    total = len(rows)
    errors = total - counts[prediction]

    return {
        "leaf": True,
        "prediction": prediction,
        "count": total,
        "errors": errors,
        "counts": counts
    }


def build_tree(rows, attributes, default_class, depth, max_depth):
    """Recursively build the pre-pruned MyDT+ decision tree."""
    if len(rows) == 0:
        return make_leaf(rows, default_class)

    if all_same_class(rows):
        return make_leaf(rows, default_class)

    if len(attributes) == 0:
        return make_leaf(rows, default_class)

    # Pre-pruning: stop before the tree becomes too specific to training data.
    if depth >= max_depth:
        return make_leaf(rows, default_class)

    current_majority = majority_class(rows, default_class)
    attr = best_attribute(rows, attributes)

    tree = {
        "leaf": False,
        "attribute": attr,
        "majority": current_majority,
        "count": len(rows),
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
            current_majority,
            depth + 1,
            max_depth
        )

    return tree


def predict(tree, row):
    """Predict one row by following matching branches in the tree."""
    if tree["leaf"]:
        return tree["prediction"]

    attr = tree["attribute"]
    value = row[attr]

    if value not in tree["children"]:
        return tree["majority"]

    return predict(tree["children"][value], row)


def train_dt_plus(rows):
    """Train MyDT+ on the supplied rows."""
    attributes = []

    for i in range(len(rows[0]) - 1):
        attributes.append(i)

    return build_tree(
        rows,
        attributes,
        majority_class(rows),
        0,
        2
    )


def classify_dt_plus(training_filename, testing_filename=None, return_tree=False):
    """Train MyDT+ and return predictions, or the tree if requested."""
    training_data = read_csv(training_filename)
    tree = train_dt_plus(training_data)

    if return_tree:
        return tree

    testing_data = read_csv(testing_filename)
    predictions = []

    for row in testing_data:
        predictions.append(predict(tree, row))

    return predictions


def format_count(number):
    return str(float(number))


def leaf_text(leaf):
    text = leaf["prediction"] + " (" + format_count(leaf["count"])

    if leaf["errors"] > 0:
        text += "/" + format_count(leaf["errors"])

    text += ")"
    return text


def print_tree_diagram(tree, indent=""):
    """Return a text-based tree diagram similar to Weka's J48 output."""
    if tree["leaf"]:
        return indent + leaf_text(tree) + "\n"

    attr_index = tree["attribute"]
    attr_name = ATTRIBUTE_NAMES[attr_index]
    result = ""

    for value in tree["children"]:
        child = tree["children"][value]

        if child["leaf"]:
            result += indent + attr_name + " = " + value + ": "
            result += leaf_text(child) + "\n"
        else:
            result += indent + attr_name + " = " + value + "\n"
            result += print_tree_diagram(child, indent + "|   ")

    return result


def count_leaves(tree):
    if tree["leaf"]:
        return 1

    total = 0

    for child in tree["children"].values():
        total += count_leaves(child)

    return total


def tree_size(tree):
    if tree["leaf"]:
        return 1

    total = 1

    for child in tree["children"].values():
        total += tree_size(child)

    return total


def evaluate_predictions(predictions, actual_labels):
    """Calculate accuracy, precision, recall, F1 and the confusion matrix."""
    total = len(actual_labels)
    correct = 0
    matrix = {}

    for actual in CLASS_LABELS:
        matrix[actual] = {}
        for predicted in CLASS_LABELS:
            matrix[actual][predicted] = 0

    for prediction, actual in zip(predictions, actual_labels):
        matrix[actual][prediction] += 1

        if prediction == actual:
            correct += 1

    class_metrics = {}

    for label in CLASS_LABELS:
        tp = matrix[label][label]
        fp = 0
        fn = 0

        for other in CLASS_LABELS:
            if other != label:
                fp += matrix[other][label]
                fn += matrix[label][other]

        if tp + fp == 0:
            precision = 0
        else:
            precision = tp / (tp + fp)

        if tp + fn == 0:
            recall = 0
        else:
            recall = tp / (tp + fn)

        if precision + recall == 0:
            f1 = 0
        else:
            f1 = 2 * precision * recall / (precision + recall)

        class_metrics[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": tp + fn
        }

    weighted_precision = 0
    weighted_recall = 0
    weighted_f1 = 0

    for label in CLASS_LABELS:
        weight = class_metrics[label]["support"] / total
        weighted_precision += class_metrics[label]["precision"] * weight
        weighted_recall += class_metrics[label]["recall"] * weight
        weighted_f1 += class_metrics[label]["f1"] * weight

    return {
        "accuracy": correct / total,
        "correct": correct,
        "incorrect": total - correct,
        "total": total,
        "matrix": matrix,
        "class_metrics": class_metrics,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1
    }


def evaluate_dt_plus_10fold(folds_filename):
    """Evaluate MyDT+ using stratified 10-fold cross-validation."""
    folds = read_folds(folds_filename)
    accuracies = []
    all_predictions = []
    all_actual_labels = []

    for i in range(10):
        testing_with_class = folds[i]
        training = []

        for j in range(10):
            if j != i:
                training.extend(folds[j])

        tree = train_dt_plus(training)
        predictions = []

        for row in testing_with_class:
            predictions.append(predict(tree, row[:-1]))

        actual_labels = []

        for row in testing_with_class:
            actual_labels.append(row[-1])

        metrics = evaluate_predictions(predictions, actual_labels)
        accuracies.append(metrics["accuracy"])
        all_predictions.extend(predictions)
        all_actual_labels.extend(actual_labels)

        print("Fold", i + 1, "accuracy:", round(metrics["accuracy"], 4))

    overall_metrics = evaluate_predictions(all_predictions, all_actual_labels)
    overall_metrics["average_fold_accuracy"] = sum(accuracies) / len(accuracies)

    return overall_metrics


def format_metric(number):
    return f"{number:.3f}"


def print_matrix_row(matrix, actual_label, letter):
    died_count = matrix[actual_label]["died"]
    survived_count = matrix[actual_label]["survived"]
    print(
        f"{died_count:4d}{survived_count:4d} |   "
        + letter
        + " = "
        + actual_label
    )


def print_evaluation_report(metrics):
    print("=== MyDT+ 10-fold cross-validation ===")
    print("Correctly Classified Instances        ", metrics["correct"],
          "             ", round(metrics["accuracy"] * 100, 4), "%")
    print("Incorrectly Classified Instances      ", metrics["incorrect"],
          "             ", round((1 - metrics["accuracy"]) * 100, 4), "%")
    print("Average fold accuracy:                ",
          round(metrics["average_fold_accuracy"], 4))
    print("Total Number of Instances             ", metrics["total"])
    print()

    print("=== Detailed Accuracy By Class ===")
    print("                 Precision  Recall   F-Measure  Class")

    for label in CLASS_LABELS:
        row = metrics["class_metrics"][label]
        print(
            "                 "
            + format_metric(row["precision"])
            + "      "
            + format_metric(row["recall"])
            + "    "
            + format_metric(row["f1"])
            + "      "
            + label
        )

    print(
        "Weighted Avg.    "
        + format_metric(metrics["weighted_precision"])
        + "      "
        + format_metric(metrics["weighted_recall"])
        + "    "
        + format_metric(metrics["weighted_f1"])
    )
    print()

    print("=== Confusion Matrix ===")
    print("   a   b   <-- classified as")
    print_matrix_row(metrics["matrix"], "died", "a")
    print_matrix_row(metrics["matrix"], "survived", "b")
    print()


def all_training_rows_from_folds(folds_filename):
    folds = read_folds(folds_filename)
    rows = []

    for fold in folds:
        rows.extend(fold)

    return rows


def print_full_training_tree(tree):
    print("=== Classifier model (full training set) ===")
    print("MyDT+ tree")
    print("------------------")
    print()
    print(print_tree_diagram(tree), end="")
    print()
    print("Number of Leaves  :\t", count_leaves(tree))
    print("Size of the tree :\t", tree_size(tree))
    print()


def write_full_training_tree(filename, tree):
    with open(filename, "w") as f:
        f.write("=== Classifier model (full training set) ===\n")
        f.write("MyDT+ tree\n")
        f.write("------------------\n\n")
        f.write(print_tree_diagram(tree))
        f.write("\n")
        f.write("Number of Leaves  :\t" + str(count_leaves(tree)) + "\n")
        f.write("Size of the tree :\t" + str(tree_size(tree)) + "\n")


def run_report(folds_filename):
    print("=== Run information ===")
    print("Scheme:       MyDT+")
    print("Relation:     heart-failure")
    print("Test mode:    10-fold cross-validation")
    print()

    metrics = evaluate_dt_plus_10fold(folds_filename)
    print()
    print_evaluation_report(metrics)

    training_rows = all_training_rows_from_folds(folds_filename)
    full_tree = train_dt_plus(training_rows)
    print("=== Appendix: Decision Tree Diagram ===")
    print()
    print_full_training_tree(full_tree)
    write_full_training_tree("mydtplus_tree.txt", full_tree)
    print("Tree diagram saved to mydtplus_tree.txt")

    return metrics


if __name__ == "__main__":
    run_report("heart-folds.csv")
