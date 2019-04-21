import csv
import random
import math
import operator
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

def loadDataset(filename, split, trainingSet=[] , testSet=[]):
	dataset = pd.read_csv(filename)
	dataset = dataset[1:100]
	for x in range(len(dataset)-1):
		if random.random() < split:
			trainingSet.append(dataset.iloc[x])
		else:
			testSet.append(dataset.iloc[x])

def euclideanDistance(instance1, instance2, length):
	distance = 0
	for x in range(length):
		distance += pow((instance1[x] - instance2[x]), 2)
	return math.sqrt(distance)

def getNeighbors(trainingSet, testInstance, k):
	distances = []
	length = len(testInstance)-1
	for x in range(len(trainingSet)):
		dist = euclideanDistance(testInstance, trainingSet[x], length)
		distances.append((trainingSet[x], dist))
	distances.sort(key=operator.itemgetter(1))
	neighbors = []
	for x in range(k):
		neighbors.append(distances[x][0])
	return neighbors

def getResponse(neighbors):
	classVotes = {}
	for x in range(len(neighbors)):
		response = neighbors[x][-1]
		if response in classVotes:
			classVotes[response] += 1
		else:
			classVotes[response] = 1
	# print(classVotes)
	sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
	# print(sortedVotes)
	return sortedVotes[0][0]

def getValidationSet(testSet):
	validations = []
	for x in range(len(testSet)):
		validations.append(testSet[x][-1])
	return validations

def main():
	# prepare data
	trainingSet=[]
	testSet=[]
	split = 0.80
	loadDataset('nursery_processed.csv', split, trainingSet, testSet)
	print('Train set: ' + repr(len(trainingSet)))
	print('Test set: ' + repr(len(testSet)))
	# generate predictions
	predictions=[]
	k = 5
	for x in range(len(testSet)):
		neighbors = getNeighbors(trainingSet, testSet[x], k)
		result = getResponse(neighbors)
		predictions.append(result)
		print('Predcted class:' + repr(result) + ', Actual class:' + repr(testSet[x][-1]))
	validations = getValidationSet(testSet)
	accuracy = accuracy_score(validations, predictions)
	cf_matrix = confusion_matrix(validations, predictions)
	class_report = classification_report(validations, predictions)
	print("Accuracy: " + repr(round(accuracy * 100, 2)) + "%")
	print("Confusion Matrix: ")
	print(cf_matrix)
	print("classification Report: ")
	print(class_report)

main()
