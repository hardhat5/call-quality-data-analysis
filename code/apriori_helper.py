import sys
import operator
from itertools import chain, combinations
from collections import defaultdict
from optparse import OptionParser


# Function to return subsets of given array
def subsets(arr):
    return chain(*[combinations(arr, i + 1) for i, a in enumerate(arr)])

# Function to load data and return generator with rows
def loadData(fname):
    file_iter = open(fname, 'r')
    for line in file_iter:
        line = line.strip().rstrip(',') 
        record = frozenset(line.split(','))
        yield record

# Function to generate 1-itemsets
def generateOneItemsets(data_iterator):

    transactionList = list()
    itemSet = set()
    for record in data_iterator:
        transaction = frozenset(record)
        transactionList.append(transaction)
        for item in transaction:
            itemSet.add(frozenset([item]))              
    return itemSet, transactionList

def returnItemsWithMinSupport(itemSet, transactionList, minSupport, freqSet):

        _itemSet = set()
        localSet = defaultdict(int)

        for item in itemSet:
                for transaction in transactionList:
                        if item.issubset(transaction):
                                freqSet[item] += 1
                                localSet[item] += 1

        for item, count in localSet.items():
                support = float(count)/len(transactionList)

                if support >= minSupport:
                        _itemSet.add(item)

        return _itemSet

# Function to self join a set and return a set of given length
def joinSet(itemSet, length):
        return set([i.union(j) for i in itemSet for j in itemSet if len(i.union(j)) == length])

# Function to return support of a given item
def getSupport(item, freqSet, transactionList):
        return float(freqSet[item])/len(transactionList)