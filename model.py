import numpy as np
import random


class ApproximateQLearning:
    def __init__(self, features, learning_rate, gama, action_space, epsilon=None) -> None:
        '''
        features: list of function(state, action) -> feature vector
        action_space: list of actions
        '''
        self.features = features
        self.w = np.zeros((len(features),))
        self.learning_rate = learning_rate
        self.gama = gama
        self.action_space = action_space
        self.epsilon = epsilon
        self.pre_scores = np.empty((0,))
        self.pos_scores = np.empty((0,))

    def normal_w(self):
        for i in range(len(self.w)):
            self.w[i] /= np.max(self.w)/10

    def set_scores(self, scores):
        self.pos_scores = np.array(scores)

    def QValue(self, state, action):
        sum = 0
        for i in range(len(self.features)):
            sum += (self.w[i] * self.features[i](state, action))
        return sum

    def update(self, state, action, next_state, reward):
        diff = (reward + self.gama * np.max([self.QValue(next_state, a) for a in self.action_space]) - self.QValue(state, action))
        for i in range(len(self.features)):
            self.w[i] += self.learning_rate * diff * self.features[i](state, action)
        self.normal_w()

    def get_action(self, state):
        if self.epsilon:
            if random.uniform(0, 1) < self.epsilon:
                return random.choice(self.action_space)
        return self.action_space[np.argmax([self.QValue(state, a) for a in self.action_space])]

    def save(self, path='w'):
        # if np.mean(self.pos_scores) > np.mean(self.pre_scores):
            np.save('scores', self.pos_scores)
            np.save(path, self.w)

    def load(self, path='w.npy'):
        self.pre_scores = np.load('scores.npy')
        self.w = np.load(path)