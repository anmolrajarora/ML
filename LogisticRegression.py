import csv

def read_data(filename):
    data = []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            data.append(row)
    return data

def str_to_float(dataset):
    for i in range(len(dataset)):
        for j in range(len(dataset[i])):
            dataset[i][j] = float(dataset[i][j])

def minMax(dataset):
    minMaxData = []
    for i in range( len(dataset[0]) - 1 ):
        column = []
        for j in range(len(dataset)):
            column.append( dataset[j][i] )
        min_value = min(column)
        max_value = max(column)
        minMaxData.append( [ min_value, max_value ] )
    return minMaxData

def normalization(dataset, minMaxData):
    for i in range(len(dataset)):
        for j in range( len(dataset[0]) -1 ):
            num = dataset[i][j]  - minMaxData[j][0]
            den = minMaxData[j][1] - minMaxData[j][0]
            dataset[i][j] = num / den

def predict():
    pass

def accuracyScore():
    pass

def stochasticGradient():
    pass

def crossValidation():
    pass

def logisticRegression():
    pass

def evaluateAlgorithm():
    pass

filename = "pima-indians-diabetes.data.csv"
dataset = read_data(filename)
str_to_float(dataset)
minMaxData = minMax(dataset)
print(minMaxData)
normalization(dataset, minMaxData)
print( dataset[:5] )
