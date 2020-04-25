import numpy as np
import scipy.spatial
from collections import Counter

class KNN:

    def __init__(self, num_clusters):
        self.num_clusters = num_clusters
        self.X_train = []
        self.y_train = []
        self.X_test = []
        
        
    def fit(self, X, y):
       
        self.X_train = X
        self.y_train = y
        
    def distance(self, a, b):
       
        distance = scipy.spatial.distance.euclidean(a, b)
    
    def predict(self, X_test):
        
        final_output = []

        for i in range(len(X_test)):
            d = []
            counts = []
            n = len(self.X_train)
            # print(self.X_train)
            for j in range(n):
                dist = scipy.spatial.distance.euclidean(self.X_train[j] , X_test[i])
                d.append([dist, j])
            
            d.sort()
            d = d[0:self.num_clusters]
            
            for d, j in d:
                counts.append(self.y_train[j])
            

            ans = Counter(counts).most_common(1)[0][0]
            final_output.append(ans)
            
        return final_output
    
    def score(self, X_test, y_test):
        
        predictions = self.predict(X_test)
        test_score = (predictions == y_test).sum() / len(y_test)
        
        return test_score

