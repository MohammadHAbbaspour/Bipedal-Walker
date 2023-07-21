import gymnasium as gym
import numpy as np
import random


env = gym.make('BipedalWalker-v3', render_mode = 'human')


QTable = {}

episode_count = 50
gamma = 0.99
alpha = 0.01
default_epsilon = 0.1
max_reward = -100

step = 0.5
action_space = []


act_1, act_2, act_3, act_4 = -1, -1, -1, -1
def buildActionSpace(): # filling action_space with all possible actions
    global action_space

    for i in range(5):
        for j in range(5):
            for k in range(5):
                for m in range(5):
                    action_space.append((act_1 + i * step, act_2 + j * step, act_3 + k * step, act_4 +  m * step))



def updateQTable(state, action, reward, nextState):
    maxQ = -200
    for a in action_space:
        new_Q = QTable.get((nextState, a), 0)
        if new_Q > maxQ:
            maxQ = new_Q
    sample_value = reward + gamma * maxQ # get sample value
    QTable[(state, action)] = (1 - alpha) * QTable.get((state, action), 0) + alpha * sample_value # update QTable



def getAction(epsilon, state):
    if random.uniform(0, 1) < epsilon: # exploration
        action = []
        for _ in range(4): 
            action.append(random.choice([-1, -0.5, 0, 0.5, 1])) # make random value from list [-1, -0.5, 0, 0.5, 1]
        action = tuple(action)
    else: # exploitation
        action = None
        max_reward = -100
        for a in action_space:
            new_val = QTable.get((state, a), 0)
            if new_val > max_reward:
                max_reward = new_val
                action = a

    return action



def getDiscreteValue(value): # value is a state
    discreteForm = []

    for i in range(len(value)):
        discreteForm.append(int(value[i])) # round to an integer

    return tuple(discreteForm)



def runNextEpisode(idx):
    global max_reward

    state = getDiscreteValue(env.reset()[0][0:14])
    total_reward = 0
    # current_epsilon = default_epsilon / (idx * 0.004)

    while True:
        next_action = getAction(default_epsilon, state) # get new action
        observation, reward, terminated, truncated, info = env.step(next_action) # perform action
        nextState = getDiscreteValue(observation[0:14].tolist()) # get discrete form of state
        total_reward += reward
        updateQTable(state, next_action, reward, nextState) 
        state = nextState
        if terminated or truncated:
            break

    if total_reward > max_reward:
        max_reward = total_reward

    return total_reward



# def plotGraph():
#     pass



def storeData():
    np.save('train_data', np.array(QTable))



def readData():
    p = np.load('train_data.npy', allow_pickle=True)
    QTable.update(p.item())



def main():
    global env
    try:
        readData()
    except:
        pass
    buildActionSpace()

    # env = gym.wrappers.RecordVideo(env, 'video', step_trigger = lambda x : x <= 1000, name_prefix='output', video_length=1000)

    for idx in range(1, episode_count + 1):
        total_reward = runNextEpisode(idx)
        print('Episode #' + str(idx) + '  =>  total_reward: ' + str(total_reward))

    print('\nHighest reward achieved: ' + str(max_reward))
    storeData()




if __name__ == '__main__':
    main()