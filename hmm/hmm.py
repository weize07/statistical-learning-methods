#encoding=utf-8

import numpy as np

class HMM(object):
    def __init__(self, ob, S, O):
        self.S = S
        self.O = O
        self.ob = ob
        # status transfer matrix;           status i -> status j
        self.A = np.zeros((self.S, self.S))  
        # status observation matrix;        status i -> observe j       
        self.B = np.zeros((self.S, self.O))   
        # init status distribute vector     PIE[i] : prob of status i        
        self.PIE = np.zeros(self.S)         

    def forward(self, ob):
        """ Cal probability of Observation recursively, in forward direction."""
        T = len(ob)
        a = np.zeros((T, self.S))
        for t in range(T):
            for i in range(self.S):
                if t == 0:
                    a[t][i] = self.PIE[i] * self.B[i][ob[t]]
                else:
                    prob = 0
                    for j in range(self.S):
                        prob += a[t - 1][j] * self.A[j][i] * self.B[i][ob[t]]
                    a[t][i] = prob
        return a
        # res = 0
        # for i in range(self.S):
        #     res += a[T - 1][i]
        # return res

    def backward(self, ob):
        """ Cal probability of Observation recursively, in backward direction. """
        T = len(ob)
        b = np.zeros((T, self.S))
        for t in range(T - 1, -1, -1):
            for i in range(self.S):
                if t == T - 1:
                    b[t][i] = 1
                else:
                    prob = 0
                    for j in range(self.S):
                        prob += b[t + 1][j] * self.A[i][j] * self.B[j][ob[t + 1]]
                    b[t][i] = prob
        return b
        # res = 0
        # for i in range(self.S):
        #     res += self.PIE[i] * self.B[i][ob[0]] * b[0][i]
        # return res

    def predict(self, ob):
        T = len(ob)
        alpha = self.forward(ob)
        res = 0
        for i in range(self.S):
            res += alpha[T - 1][i]
        return res

    def predict_b(self, ob):
        T = len(ob)
        beta = self.backward(ob)
        res = 0
        for i in range(self.S):
            res += self.PIE[i] * self.B[i][ob[0]] * beta[0][i]
        return res

    def train(self):
        """ Estimate matrix A,B and vector PIE, using training dataset. """
        self._banum_welch(self.ob)
        # self.A = np.array([[0.5, 0.2, 0.3],[0.3, 0.5, 0.2],[0.2, 0.3, 0.5]])
        # self.B = np.array([[0.5, 0.5],[0.4, 0.6],[0.7, 0.3]])
        # self.PIE = np.array([0.2, 0.4, 0.4])

    def _banum_welch(self, ob):
        """ A special case of EM algorithm, 
        to make maxium likelihood estimation of params."""
        T = len(ob)
        self.A = np.array([[0.1, 0.6, 0.3],[0.3, 0.5, 0.2],[0.2, 0.3, 0.5]])
        self.B = np.array([[0.1, 0.9],[0.4, 0.6],[0.7, 0.3]])
        self.PIE = np.array([0.2, 0.3, 0.5])
        for count in range(100):
            alpha = self.forward(ob)
            beta = self.backward(ob)
            gammas = np.zeros((T, self.S))
            epsilons = np.zeros((T, self.S, self.S))
            for t in range(T):
                for i in range(self.S):
                    gammas[t][i] = self._gamma(t, i, alpha, beta)
                    if t == T - 1:
                        continue
                    for j in range(self.S):
                        epsilons[t][i][j] = self._epsilon(t, i, j, alpha, beta, ob)
            for i in range(self.S):
                denominator = 0
                for t in range(T - 1):
                    denominator += gammas[t][i]
                for j in range(self.S):
                    numerator = 0
                    for t in range(T - 1):
                        numerator += epsilons[t][i][j]
                    self.A[i][j] = numerator / denominator
            for i in range(self.S):
                denominator = 0
                for t in range(T):
                    denominator += gammas[t][i]
                for k in range(self.O):
                    numerator = 0
                    for t in range(T):
                        if (ob[t] == k):
                            numerator += gammas[t][i]
                    self.B[i][k] = numerator / denominator
            for i in range(self.S):
                self.PIE[i] = gammas[0][i]

        return 

    def _gamma(self, t, i, alpha, beta):
        prob_sum = 0
        for j in range(self.S):
            prob_sum += alpha[t][j] * beta[t][j]
        return alpha[t][i] * beta[t][i] / prob_sum


    def _epsilon(self, t, i, j, alpha, beta, ob):
        prob_sum = 0
        for ii in range(self.S):
            for jj in range(self.S):
                prob_sum += alpha[t][ii] * self.A[ii][jj] * self.B[jj][ob[t+1]] * beta[t+1][jj]
        return alpha[t][i] * self.A[i][j] * self.B[j][ob[t+1]] * beta[t+1][j] / prob_sum


def main():
    hmm = HMM([0, 1, 0], 3, 2)
    hmm.train()

    print hmm.A
    print hmm.B
    print hmm.PIE

    print hmm.predict(np.array([0, 1, 0]))
    print hmm.predict_b(np.array([0, 1, 0]))

if __name__ == '__main__':
    main()




