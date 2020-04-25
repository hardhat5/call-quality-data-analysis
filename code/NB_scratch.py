import pandas as pd
import numpy as np

# Count of observations with ycol and xcol
def cnt(x,y,xcol,ycol):
    tempx,tempy = x[y == ycol], y[y == ycol]
    tempx,tempy = tempx[x[xcol] == 1], tempy[x[xcol] == 1]
    return len(tempy)

# Filling in conditional probabilities P(explantory_attribute|predction_labels)

def fillProb(X_tr, Y_tr, cntCDC, probCDC):
    # Conditional probabilities P(X_column|Call Dropped),...
    probabilities = {'Call Dropped': {},'Poor Network': {},'Poor Voice Quality': {},'Satisfactory': {}}
    for col in list(X_tr.columns):
        probabilities['Call Dropped'][col] = {}
        probabilities['Poor Voice Quality'][col] = {}
        probabilities['Poor Network'][col] = {}
        probabilities['Satisfactory'][col] = {}
    
        cnt_CallDropped = cnt(X_tr, Y_tr,col,'Call Dropped')
        cnt_PoorVoiceQuality = cnt(X_tr, Y_tr,col,'Poor Voice Quality')
        cnt_PoorNetwork = cnt(X_tr, Y_tr,col,'Poor Network')
        cnt_Satisfactory = cnt(X_tr, Y_tr,col,'Satisfactory')
    
        if (probCDC['Call Dropped'] != 0):
            probabilities['Call Dropped'][col] = cnt_CallDropped/cntCDC['Call Dropped']
        else:
            probabilities['Call Dropped'][col] = 0
        if (probCDC['Poor Voice Quality'] != 0):
            probabilities['Poor Voice Quality'][col] = cnt_PoorVoiceQuality/cntCDC['Poor Voice Quality']
        else:
            probabilities['Poor Voice Quality'][col] = 0
        if (probCDC['Poor Network'] != 0):
            probabilities['Poor Network'][col] = cnt_PoorNetwork/cntCDC['Poor Network']
        else:
            probabilities['Poor Network'][col] = 0
        if (probCDC['Satisfactory'] != 0):
            probabilities['Satisfactory'][col] = cnt_Satisfactory/cntCDC['Satisfactory']
        else:
            probabilities['Satisfactory'][col] = 0

    return probabilities


class NBClassify:
    def __init__(self):
        
        self.Xtrain = []
        self.Ytrain = []
        self.Y_labels = []

        # self.Xtrain = X1
        # self.Ytrain = Y1
        # self.Y_labels = ['Call Dropped','Poor Voice Quality','Poor Network','Satisfactory']
        
    def fitNB(self,X,Y):
        # cnt_CDC = count of each classification variables
        # prob_CDC = probability of each classification
        
        self.Xtrain = X
        self.Ytrain = Y
        # self.Y_labels = ['Call Dropped','Poor Voice Quality','Poor Network','Satisfactory']
        self.Y_labels = np.unique(self.Ytrain)

        self.cnt_CDC = {y: sum(self.Ytrain == y) for y in self.Y_labels}
        self.prob_CDC = {y: self.cnt_CDC[y]/len(self.Ytrain) for y in self.Y_labels}
        self.probabilities = fillProb(self.Xtrain,self.Ytrain,self.cnt_CDC,self.prob_CDC)
    
    def predict(self,X1):
        # predicts: classification results
        predicts = []
        for r in range(0,len(X1)):
            prd = {y: self.prob_CDC[y] for y in self.Y_labels}
            for n in range(0,len(X1.columns)):
                if (X1.iloc[r,n] == 0):
                    continue
                else:
                    for y in self.Y_labels:
                        prd[y] *= self.probabilities[y][X1.columns[n]]
            predicts.append(max(prd, key=prd.get))
        return predicts
    
    def score(self,pred,Y1):
        # pred: predictions generated
        # Y1: test observations true values
        res_cnt = 0
        for i in range(0,len(Y1)):
            if (Y1.iloc[i] == pred[i]):
                res_cnt = res_cnt+1
        return res_cnt/len(Y1)
