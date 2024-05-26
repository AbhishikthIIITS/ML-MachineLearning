import csv
import numpy as np

print('\nName:- Abhishikth Boda')
print('Roll Number:- S20210010044')
print('Course:- Machine Learning')
print('Section:- 2')
print('Assignment Number:- 1')
print('Date:- 10th August 2023\n')

print('Program is being executed\n')

def load_data(file_path):
    data = []
    labels = []
    with open(file_path, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')  
        next(reader)  
        for row in reader:
            data.append(list(map(float, row[:-1])))
            labels.append(int(row[-1]))
    return np.array(data), np.array(labels)

def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

def find_neighbors(train_data, train_labels, test_point, k):
    distances = [euclidean_distance(test_point, train_point) for train_point in train_data]
    sorted_indices = np.argsort(distances)
    k_nearest_labels = train_labels[sorted_indices[:k]]
    return k_nearest_labels

def predict_class(neighbors):
    unique, counts = np.unique(neighbors, return_counts=True)
    index = np.argmax(counts)
    return unique[index]

def normalize_data(data):
    min_vals = np.min(data, axis=0)
    max_vals = np.max(data, axis=0)
    normalized_data = (data - min_vals) / (max_vals - min_vals)
    return normalized_data

def calculate_mean(data):
    return np.mean(data, axis=0)

def calculate_std(data):
    return np.std(data, axis=0)

def manhattan_distance(point1, point2):
    return np.sum(np.abs(point1 - point2))

def shuffle_data(data, labels):
    shuffled_indices = np.random.permutation(len(data))
    shuffled_data = data[shuffled_indices]
    shuffled_labels = labels[shuffled_indices]
    return shuffled_data, shuffled_labels

def train_validation_split(data, labels, validation_ratio=0.2):
    num_validation = int(len(data) * validation_ratio)
    train_data = data[num_validation:]
    train_labels = labels[num_validation:]
    validation_data = data[:num_validation]
    validation_labels = labels[:num_validation]
    return train_data, train_labels, validation_data, validation_labels

def calculate_accuracy(predictions, true_labels):
    correct = np.sum(predictions == true_labels)
    total = len(true_labels)
    accuracy = correct / total
    return accuracy

def correct_predictions(predictions, true_labels):
    correct = np.sum(predictions == true_labels)
    return correct

def total_predictions(predictions, true_labels):
    total = len(true_labels)
    return total

train_data, train_labels = load_data('winequality-white-Train.csv')
test_data, test_labels = load_data('winequality-white-Test.csv')

k_1_predictions = []
for test_point in test_data:
    neighbors = find_neighbors(train_data, train_labels, test_point, 1)
    predicted_label = predict_class(neighbors)
    k_1_predictions.append(predicted_label)
k_1_accuracy = calculate_accuracy(k_1_predictions, test_labels)

k_3_predictions = []
for test_point in test_data:
    neighbors = find_neighbors(train_data, train_labels, test_point, 3)
    predicted_label = predict_class(neighbors)
    k_3_predictions.append(predicted_label)
k_3_accuracy = calculate_accuracy(k_3_predictions, test_labels)
correct_1 = correct_predictions(k_1_predictions, test_labels)
total_1 = total_predictions(k_1_predictions, test_labels)
correct_3 = correct_predictions(k_3_predictions, test_labels)
total_3 = total_predictions(k_3_predictions, test_labels)

with open('S2021001004__1-NNC_Accuracy_Report.txt', 'w') as f:
    f.write(f'Name:- Abhishikth Boda\n')
    f.write(f'Roll Number:- S20210010044\n')
    f.write(f'Course:- Machine Learning\n')
    f.write(f'Section:- 2\n')
    f.write(f'Assignment Number:- 1\n')
    f.write(f'Date:- 10th August 2023\n\n\n')

    f.write(f'1-Nearest Neighbor Classifier (1-NNC)\n\n')

    f.write(f'1-NNC Accuracy: {k_1_accuracy:.6f}\n')
    f.write(f'1-NNC Correct Predictions: {correct_1}\n')
    f.write(f'1-NNC Total Predictions: {total_1}\n\n')
    

with open('S2021001004__3-NNC_Accuracy_Report.txt', 'w') as f:
    f.write(f'Name:- Abhishikth Boda\n')
    f.write(f'Roll Number:- S20210010044\n')
    f.write(f'Course:- Machine Learning\n')
    f.write(f'Section:- 2\n')
    f.write(f'Assignment Number:- 1\n')
    f.write(f'Date:- 10th August 2023\n\n\n')
    
    f.write(f'3-Nearest Neighbor Classifier (3-NNC)\n\n')
    f.write(f'3-NNC Accuracy: {k_3_accuracy:.6f}\n')
    f.write(f'3-NNC Correct Predictions: {correct_3}\n')
    f.write(f'3-NNC Total Predictions: {total_3}\n')


