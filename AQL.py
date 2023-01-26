import gymnasium as gym
import numpy as np
import random
from model import ApproximateQLearning
from draw import Plot

env = gym.make('BipedalWalker-v3', render_mode = 'human')
training_episodes = 1000

part_size = 0.15
upper_bound = 1
lower_bound = -1
part_number = (upper_bound - lower_bound)/part_size
actions_idx = [i for i in range(int(part_number))]
action_space = [[-1 for j in range(4)] for i in range(int(part_number) ** 4)]
k = 50

count = 0
for i in range(int(part_number)):
    for j in range(int(part_number)):
        for u in range(int(part_number)):
            for v in range(int(part_number)):
                action_space[count][0] = actions_idx[i]
                action_space[count][1] = actions_idx[j]
                action_space[count][2] = actions_idx[u]
                action_space[count][3] = actions_idx[v]
                count += 1


def generate_action(action_idx):
    lowers = [lower_bound + part_size * action_idx[i] for i in range(4)]
    uppers = [lower_bound + part_size * (action_idx[i] + 1) for i in range(4)]
    return [random.uniform(lowers[i], uppers[i]) for i in range(4)]

def hull_x_speed(state, action):
    action = generate_action(action)
    return state[1]/np.cos(state[0])

def hull_y_speed(state, action):
    action = generate_action(action)
    return state[1]/np.sin(state[0])

def hip_1_x_speed(state, action):
    action = generate_action(action)
    return (state[5] + action[0])/np.cos(state[4])

def hip_1_y_speed(state, action):
    action = generate_action(action)
    return (state[5] + action[0])/np.sin(state[4])

def hip_2_x_speed(state, action):
    action = generate_action(action)
    return (state[10] + action[2])/np.cos(state[9])

def hip_2_y_speed(state, action):
    action = generate_action(action)
    return (state[10] + action[2])/np.sin(state[9])

def knee_1_x_speed(state, action):
    action = generate_action(action)
    return (state[7] + action[1])/np.cos(state[6])

def knee_1_y_speed(state, action):
    action = generate_action(action)
    return (state[7] + action[1])/np.sin(state[6])

def knee_2_x_speed(state, action):
    action = generate_action(action)
    return (state[12] + action[3])/np.cos(state[11])

def knee_2_y_speed(state, action):
    action = generate_action(action)
    return (state[12] + action[3])/np.sin(state[11])

def vel_x_speed(state, action):
    return state[2]

def vel_y_speed(state, action):
    return state[3]

def check_contact_with_ground(state, action):
    global k
    if state[8] and state[13]:
        return -3*k
    elif (state[8] and not state[13]) or (state[13] and not state[8]):
        return 6*k
    return -3*k

features = [hull_x_speed, hull_y_speed, hip_1_x_speed, hip_1_y_speed, hip_2_x_speed, hip_2_y_speed, 
            knee_1_x_speed, knee_1_y_speed, knee_2_x_speed, knee_2_y_speed, vel_x_speed, vel_y_speed, check_contact_with_ground]

aql = ApproximateQLearning(features, 0.5, 1, action_space, 0.2)
aql.load()
state = env.reset()[0]

scores = []
time = []
total_reward = 0
env = gym.make("BipedalWalker-v3", render_mode='rgb_array')
env = gym.wrappers.RecordVideo(env, 'video2', step_trigger = lambda x: x<= 2*training_episodes, name_prefix='output', video_length=240, disable_logger=True)
env.reset()
for episode in range(training_episodes):
    action = aql.get_action(state)
    next_state, reward, terminated, truncated, info = env.step(generate_action(action))
    env.render()
    total_reward += reward
    scores.append(total_reward)
    if terminated or truncated:
        total_reward = 0
        reward = -100
        env.reset()
    aql.update(state, action, next_state, reward)
    print(aql.w.tolist())
    state = next_state
    time.append(episode)
aql.set_scores(scores)
aql.save()
print('Max score: ', max(scores), ' -- Avg score: ', sum(scores)/len(scores))
env.close()

Plot(time, scores).draw2d('plot.png')