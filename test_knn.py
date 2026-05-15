import csv

# MyKNN is the baseline k-nearest neighbour classifier. It treats every
# selected neighbour equally during voting.

def classify_knn(training_filename, testing_filename, k):
    """Classify each testing row using unweighted k-nearest neighbours."""

    training_data = []
    testing_data = []
    result = []

    with open(training_filename, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            training_data.append(row)

    with open(testing_filename, "r") as f:
        reader = csv.reader(f)
        testing_data = list(reader)

    for row in testing_data:
        distances = []

        # Use simple categorical distance: one point for each differing
        # attribute value.
        for index, data in enumerate(training_data):
            distance = 0

            for i in range(len(row)):
                if row[i] != data[i]:
                    distance += 1
        
            distances.append([distance, index, data[-1]])

        # Sort by distance, then by original training order for stable ties.
        distances.sort(key=lambda x: (x[0], x[1]))

        distances = distances[:k]

        died = 0
        survived = 0

        for distance in distances:
            if distance[-1] == 'died':
                died += 1
            else:
                survived += 1

        if survived > died:
            result.append('survived')
        
        else:
            result.append('died')

    return result

def read_folds(folds_filename):
    """Read the assignment fold file into ten stratified folds."""

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
    """Write rows to a temporary CSV file."""

    with open(filename, "w", newline="") as f:

        writer = csv.writer(f)

        writer.writerows(rows)

def evaluate_knn_10fold(folds_filename, k):
    """Evaluate MyKNN with stratified 10-fold cross-validation."""

    folds = read_folds(folds_filename)

    accuracies = []

    for i in range(10):

        testing_with_class = folds[i]

        training = []

        for j in range(10):

            if j != i:

                training.extend(folds[j])

        # The classifier should predict the class, so remove it from testing.
        testing_without_class = []

        for row in testing_with_class:

            testing_without_class.append(row[:-1])

        write_csv("train_temp.csv", training)

        write_csv("test_temp.csv", testing_without_class)

        predictions = classify_knn("train_temp.csv", "test_temp.csv", k)

        correct = 0

        for prediction, actual_row in zip(predictions, testing_with_class):

            actual_class = actual_row[-1]

            if prediction == actual_class:

                correct += 1

        accuracy = correct / len(testing_with_class)

        accuracies.append(accuracy)

        #print("Fold", i + 1, "accuracy:", accuracy)

    average_accuracy = sum(accuracies) / len(accuracies)

    print("Average accuracy:", average_accuracy)

    return average_accuracy

evaluate_knn_10fold("heart-folds.csv", 1)
evaluate_knn_10fold("heart-folds.csv", 2)
evaluate_knn_10fold("heart-folds.csv", 3)
evaluate_knn_10fold("heart-folds.csv", 4)
evaluate_knn_10fold("heart-folds.csv", 5)
evaluate_knn_10fold("heart-folds.csv", 6)
