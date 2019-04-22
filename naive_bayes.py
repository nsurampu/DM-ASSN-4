import numpy as np 
import pandas as pd 
from pprint import pprint

data = pd.read_csv('nursery.csv',header=None)
data = data.values

attribute_names = data[0,:]
features = data[1:,:-1]
labels = data[1:,-1]
unique_values = list(set(labels.tolist()))
unique_values.sort()
num_classes = len(unique_values)

unique_values = {key.strip():unique_values.index(key) for key in unique_values}
print(unique_values)
train_fraction = 0.8

train_size = int(features.shape[0]*train_fraction)

train_features = features[:train_size,:].tolist()
train_labels = labels[:train_size].tolist()

test_features = features[train_size:,:].tolist()
test_labels = labels[train_size:].tolist()



conditional_probability_table = {}

for row_index in range(len(train_features)):
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



accuracy = 0

predicted_matrix = [0 for i in range(num_classes)]
actual_matrix = [0 for i in range(num_classes)]

for row_index in range(len(test_features)):
    result = [1 for i in range(num_classes)]
    for attribute_index in range(len(test_features[row_index])):
        key = attribute_names[attribute_index] + ' : ' + test_features[row_index][attribute_index]
        result = [a*b for a,b in zip(conditional_probability_table[ key],result)]
    result = np.argmax(result)
    actual_index = unique_values[test_labels[row_index]]
    predicted_matrix[result] += 1
    actual_matrix[actual_index] += 1
    if actual_index == result:
        accuracy += 1


print(accuracy/len(test_features))
print(predicted_matrix)
print(actual_matrix)
print(len(test_features),train_size)