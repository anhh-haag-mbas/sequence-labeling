import numpy as np

class CrfLayer:
    def __init__(self, num_labels):
        self.num_labels = num_labels
        self.transitions = np.random.rand(num_labels + 2, num_labels + 2)
        self.transitions[(0, -1), :] = -1000
        self.transitions[0, -2] = 0
        self.transitions[-1, -1] = 0
        print(self.transitions)

    # Implementation inspired by: https://createmomo.github.io/2017/11/24/CRF-Layer-on-the-Top-of-BiLSTM-6/ 
    def viterbi(self, probability_distributions):
        alpha = ([],[])
        path = []
        transitions = self.transitions[1:1 + self.num_labels, 1:1 + self.num_labels]

        previous = probability_distributions[0, :] 
        for i in range(len(probability_distributions)):
            obs = probability_distributions[i, :] 
            
            # 1. Expand the previous
            previous = np.resize(previous, (self.num_labels, len(previous)))
            previous = np.transpose(previous)

            # 2. Expand the obs
            obs = np.resize(obs, (self.num_labels, len(obs)))
        
            # 3. Sum previous, obs and transition scores
            scores = previous + obs + transitions 

            # Change previous for next iteration 
            previous = np.amax(scores, axis = 0)

            # Add best scores and column indexes into alpha
            alpha[0].append(previous)
            alpha[1].append(scores.argmax(axis = 0)) # Index of highest value in each column
 
        path = []

        # Find the best path by moving backwards 
        idx = np.argmax(alpha[0][-1])
        for i in range(len(alpha[0])):
            path.append(idx)
            idx = alpha[1][-(1 + i)][idx]
        path.append(idx)

        return path[::-1]


    def total_loss(self):
        pass































