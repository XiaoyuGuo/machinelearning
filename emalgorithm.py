# Expectation Maximization
# Author  : Guoxiaoyu
# Date    : 2017/3/3
# Version : Python3.6.0
# An Example of Expectation Maximization

class EMAlgorithm():
    '''EM algorithm'''

    def __init__(self, event_seq):
        self.init_probs = [0.3, 0.4, 0.5] # A , B , C == 1
        self.event_seq = event_seq

    def applyEM(self):
        '''Expectation Maxization'''

        prob_from_B = []
        
        # Travel whole ob seq
        for ob_elem in self.event_seq:
            # P(A -> 1 -> B -> 1) / P(A -> 1 -> B -> 1) + P(A -> 0 -> C -> 1)
            top = self.init_probs[0] * pow(self.init_probs[1], ob_elem) * pow((1 - self.init_probs[1]), 1 - ob_elem)
            bottom = top + (1 - self.init_probs[0]) * pow(self.init_probs[2], ob_elem) * pow((1 - self.init_probs[2]), 1 - ob_elem)
            prob_from_B.append(top/bottom)

        # Update P(A -> 1)
        self.init_probs[0] = sum(prob_from_B) / len(prob_from_B)
        
        # Update P(B -> 1)
        top = 0
        for i in range(len(self.event_seq)):
            top += self.event_seq[i] * prob_from_B[i]

        self.init_probs[1] = top / sum(prob_from_B)

        # Update P(C -> 1)
        top = 0
        bottom = 0
        for i in range(len(self.event_seq)):
            top += (1 - prob_from_B[i]) * self.event_seq[i]
            bottom += (1 - prob_from_B[i])
        
        self.init_probs[2] = top / bottom
    
    def iter(self, n):
        for i in range(n):
            self.applyEM()

event_seq = [1, 1, 0, 1, 0, 0, 1, 0, 1, 1]
a = EMAlgorithm(event_seq)
a.iter(1000)
print(a.init_probs)