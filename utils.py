import numpy as np

def excluding_None(values):
    return [value for value in values if value != None]

def statistics_excluding_None(values, statistics_function):
    values_excluding_None = excluding_None(values)
    if len(values_excluding_None) > 0:
        statistics = statistics_function(values_excluding_None)
        return np.round(statistics, decimals=3)
    else:
        return None

def mean(values):
    return statistics_excluding_None(values, np.mean)
        
def std(values):
    return statistics_excluding_None(values, np.std)
    
def max(values):
    return statistics_excluding_None(values, np.max)
    
def min(values):
    return statistics_excluding_None(values, np.min)

def count(values):
    return len(excluding_None(values))

def read_texts(path):
    texts = list()
    with open(path, 'r', encoding='utf-8', errors='ignore') as file:
        for line in file:
            texts.append(line.strip())
    return texts
