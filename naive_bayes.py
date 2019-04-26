import numpy as np 
import pandas as pd 
from pprint import pprint
from sklearn.metrics import confusion_matrix

np.random.seed(100)

'nursery.csv'
def naive_bayes(path_to_file):
    
    """ Description
    :type path_to_file: str
    :param path_to_file: the path to the .csv file

    :rtype: tuple: the predicted classes and the actual classs
    """
    data = pd.read_csv(path_to_file,header=None)
    data = data.values
    attribute_names = data[0,:]
    data = data[1:,:]
    np.random.shuffle(data)
    features = data[1:,:-1]
    labels = data[1:,-1]
    unique_values = list(set(labels.tolist()))
    unique_values.sort()
    num_classes = len(unique_values)

    unique_values = {key.strip():unique_values.index(key) for key in unique_values}
    train_fraction = 0.8

    train_size = int(features.shape[0]*train_fraction)

    train_features = features[:train_size,:].tolist()
    train_labels = labels[:train_size].tolist()

    test_features = features[train_size:,:].tolist()
    test_labels = labels[train_size:].tolist()



    conditional_probability_table = {}
    prior_distribution = {key:0 for key in unique_values.keys()}

    for row_index in range(len(train_features)):
        prior_distribution[train_labels[row_index]] += 1
        for attribute_index in range(len(train_features[row_index])):
            key = attribute_names[attribute_index] + ' : ' + train_features[row_index][attribute_index]
            # print(key,train_labels[row_index],unique_values[train_labels[row_index]])
            if key not in conditional_probability_table:
                conditional_probability_table[key] = [0 for i in range(num_classes)]
                index = unique_values[train_labels[row_index]]
                conditional_probability_table[key][index] = 1
            else:
                index = unique_values[train_labels[row_index]]
                conditional_probability_table[key][index] += 1

    for key in conditional_probability_table.keys():
        conditional_probability_table[key] = [item/train_size for item in conditional_probability_table[key]]

    for key in prior_distribution.keys():
        prior_distribution[key] /= train_size


    accuracy = 0

    predicted_matrix = [0 for i in range(num_classes)]
    actual_matrix = [0 for i in range(num_classes)]


    pred_labels = []
    actual_labels = []
    for row_index in range(len(test_features)):
        result = [prior_distribution[i] for i in prior_distribution.keys()]
        for attribute_index in range(len(test_features[row_index])):
            key = attribute_names[attribute_index] + ' : ' + test_features[row_index][attribute_index]
            result = [a*b for a,b in zip(conditional_probability_table[ key],result)]
        result = np.argmax(result)
        actual_index = unique_values[test_labels[row_index]]
        predicted_matrix[result] += 1
        actual_matrix[actual_index] += 1
        pred_labels.append(result)
        actual_labels.append(actual_index)
        if actual_index == result:
            accuracy += 1


    # print(prior_distribution)
    print('accuracy: ',accuracy/len(test_features)*100,'%')
    print('predicted: matrix: ',predicted_matrix)
    print('actual matrix: ',actual_matrix)
    # print(len(test_features),train_size)
    return confusion_matrix(actual_labels,pred_labels)

if __name__ == "__main__":
    print('confusion matrix: \n',naive_bayes('nursery.csv'))