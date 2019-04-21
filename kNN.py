import csv
import random
import math
import operator
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

class KNN:
	def load_dataset(self, filename, split, training_set=[] , validation_set=[]):
		dataset = pd.read_csv(filename)
		dataset = dataset[1:100]
		for x in range(len(dataset)-1):
			if random.random() < split:
				training_set.append(dataset.iloc[x])
			else:
				validation_set.append(dataset.iloc[x])

	def distance_measure(self, instance1, instance2, length):
		distance = 0
		for x in range(length):
			distance += pow((instance1[x] - instance2[x]), 2)
		return math.sqrt(distance)

	def neighbors(self, training_set, val_instance, k):
		distances = []
		length = len(val_instance)-1
		for x in range(len(training_set)):
			dist = self.distance_measure(val_instance, training_set[x], length)
			distances.append((training_set[x], dist))
		distances.sort(key=operator.itemgetter(1))
		neighbors = []
		for x in range(k):
			neighbors.append(distances[x][0])
		return neighbors

	def get_predictions(self, neighbors):
		class_count = {}
		for x in range(len(neighbors)):
			response = neighbors[x][-1]
			if response in class_count:
				class_count[response] += 1
			else:
				class_count[response] = 1
		# print(class_count)
		sorted_counts = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
		# print(sorted_counts)
		return sorted_counts[0][0]

	def get_validations(self, validation_set):
		validations = []
		for x in range(len(validation_set)):
			validations.append(validation_set[x][-1])
		return validations

if __name__ == "__main__":
	knn = KNN()

	training_set=[]
	validation_set=[]
	split = 0.80
	knn.load_dataset('nursery_processed.csv', split, training_set, validation_set)
	print("Train-Test Split: " + repr(split * 100) + "-" + repr(round((1-split),1) * 100))
	print("Train set: " + repr(len(training_set)))
	print("Test set: " + repr(len(validation_set)))
	# generate predictions
	predictions=[]
	k = 5
	for x in range(len(validation_set)):
		neighbors = knn.neighbors(training_set, validation_set[x], k)
		result = knn.get_predictions(neighbors)
		predictions.append(result)
		print("Predcted class: " + repr(result) + ", Actual class: " + repr(validation_set[x][-1]))
	validations = knn.get_validations(validation_set)
	accuracy = accuracy_score(validations, predictions)
	cf_matrix = confusion_matrix(validations, predictions)
	class_report = classification_report(validations, predictions)
	print("Accuracy: " + repr(round(accuracy * 100, 2)) + "%")
	print("Confusion Matrix: ")
	print(cf_matrix)
	print("classification Report: ")
	print(class_report)
