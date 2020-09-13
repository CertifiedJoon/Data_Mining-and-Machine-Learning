import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
from matplotlib import style
import warnings
from collections import Counter
import pandas as pd
import random
style.use('fivethirtyeight')
##dataset = {'k':[[1,2],[2,3],[3,1]], 'r': [[6,5],[7,7],[8,6]]}
##new_features = [5,7]
##[[plt.scatter(ii[0],ii[1], s=100,color=i) for ii in dataset[i]] for i in dataset]
##plt.scatter(new_features[0], new_features[1], s=100)
##plt.show()

def k_nearest_neighbors(data, label, k=3):
    if len(data) >= k:
        warning.warn("K is set to a value less than the total voting group")
    distances = []
    for group in data:
        for features in data[group]:
            euclidean_distance = np.linalg.norm(np.array(features) - np.array(label))
            distances.append([euclidean_distance, group])

    votes = [i[1] for i in sorted(distances[:k])]
    vote_result = Counter(votes).most_common(1)[0][0]
    return vote_result

df = pd.read_csv("breast-cancer.csv")
df.drop(['id'], 1,inplace = True)
df.dropna(axis = 1,inplace = True)
df.replace( 'M', 4, inplace = True)
df.replace('B', 2, inplace= True)
df = df.values.tolist()
test_size = 0.9

train_set = {2: [], 4: []}
test_set = {2: [], 4: []}
train_data = df[:int(test_size*len(df))]
test_data = df[int(test_size*len(df)):]
##print(train_data.iloc[:, 0])
##print(df.head())

for x in train_data:
    train_set[x[0]].append(x[1:])

for i in test_data:
    test_set[i[0]].append(i[1:])

correct = 0
total = 0

for group in test_set:
    for data in test_set[group]:
        vote = k_nearest_neighbors(train_set, data, k = 5)
        if group == vote:
            correct+=1
        total +=1;

print('Accurace: ', correct/total)
            
        

