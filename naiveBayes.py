# NaiveBayes Algorithm
# Date    : 2017/3/2
# Author  : Guoxiaoyu
# Version : Python3.6.0

import random

class LabeledVector():
    '''
    Unit of a sample
    '''
    def __init__(self, vector, label):
        self.vector = vector # vector : Tuple
        self.label  = label  # label  : Boolean

class NaiveBayes():
    '''
    NaiveBayes Algorithm
    '''
    def genSamples(self):
        '''
        Generate samples
        '''
        samples = []
        for _ in range(1000):
            samples.append(LabeledVector((True, True, False), True))
            samples.append(LabeledVector((False, True, False), False))
            samples.append(LabeledVector((False, False, False), False))
            samples.append(LabeledVector((False, False, True), False))
            samples.append(LabeledVector((True, False, False), True))
            samples.append(LabeledVector((True, False, True), True))
            samples.append(LabeledVector((False, True, True), False))
            samples.append(LabeledVector((True, True, True), True))
        return samples
    
    def calcProbOfLabels(self, samples):
        '''Calc prob of label'''

        labels_prob = {}

        for sample in samples:
            if not str(sample.label) in labels_prob:
                labels_prob[str(sample.label)] = 1
            else:
                labels_prob[str(sample.label)] += 1

        size = len(samples)

        for label in labels_prob:
            labels_prob[label] /= size

        self.labels_prob = labels_prob

    def calcProbOfFeaturesWithCondition(self, samples):
        '''Calc prob of features with condition'''
        features_prob = {}
        size = len(samples)
        for sample in samples:
            for indexOfFeature, value in enumerate(sample.vector):
                if not str(indexOfFeature)+"|"+str(sample.label) in features_prob:
                    features_prob[str(indexOfFeature)+"|"+str(sample.label)] = {}
                    features_prob[str(indexOfFeature)+"|"+str(sample.label)][value] = 1
                else:
                    if not value in features_prob[str(indexOfFeature)+"|"+str(sample.label)]:
                        features_prob[str(indexOfFeature)+"|"+str(sample.label)][value] = 1
                    else:
                        features_prob[str(indexOfFeature)+"|"+str(sample.label)][value] += 1

        for featureWithCondition in features_prob:
            for value in features_prob[featureWithCondition]:
                features_prob[featureWithCondition][value] /= self.labels_prob[featureWithCondition.split("|")[1]] * size
        self.features_prob = features_prob
    
    
    def predict(self, vector):
        prob_dict = {}
        for label in self.labels_prob:
            prob_dict[label] = self.labels_prob[label]
            for indexOfFeature, value in enumerate(vector):
                if not value in self.features_prob[str(indexOfFeature)+"|"+str(label)]:
                    prob_dict[label] *= 0
                else:
                    prob_dict[label] *= self.features_prob[str(indexOfFeature)+"|"+str(label)][value]
        print(max(prob_dict, key=lambda x:prob_dict[x]))

# Bayes Formula  P(A|B) = P(B|A) * P(A) / P(B)
# B : Vector(b1, b2, ... , bn) && P(b1, b2, b3, ... , bn) = P(b1) * P(b2) * P(b3) * ... * P(bn)
# A : Label
# Inference:
# P(B) == 1
# P(A|B) = P(B|A) * P(A) 
# P(A = True | B = (True, True, True)) == P(b1 = True | A = True) * P(b2 = True | A = True) * P(b3 = True | A = True) * P(A = True)

    def train(self, samples):
        labels_prob = self.calcProbOfLabels(samples)
        features_prob = self.calcProbOfFeaturesWithCondition(samples)
        return labels_prob, features_prob

nv = NaiveBayes()
samples = nv.genSamples()
nv.train(samples)
nv.predict((False,False,False))


    