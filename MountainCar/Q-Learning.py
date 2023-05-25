import numpy as np
from MountainCarEnv import MountainCarEnv
import statistics


class QLearning():

    def __init__(self, render):
        self.env = MountainCarEnv(render_mode = render)
        self.actions = list(range(self.env.action_space.n))
        self.rewards = []
            
    def discretizar(self):
        self.pos_space = np.linspace(-5, 5, 10)
        self.vel_space = np.linspace(-3, 3, 2)
        self.Q = np.zeros((11,3,3)) #cargar con -500

    def get_state(self, obs):
        pos, vel = obs
        pos_bin = np.digitize(pos, self.pos_space)
        vel_bin = np.digitize(vel, self.vel_space)
        return pos_bin, vel_bin

    def epsilon_greedy_policy(self, state, Q, epsilon=0.1):
        explore = np.random.binomial(1, epsilon)
        if explore:
            action = self.env.action_space.sample()
            print('explore')
        # exploit
        else:
            action = np.argmax(Q[state])
            print('exploit')
        return action

    def optimal_policy(state, Q):
        action = np.argmax(Q[state])
        return action

    def qLearning(self, iterations, alpha, epsilon, gamma):
        count = 0
        while(count < iterations):
            obs = self.env.reset()
            initial_state_Q = np.array(iterations)
            print(obs)
            done = False
            while not done:
                previousState = self.get_state(obs)
                action = self.epsilon_greedy_policy(previousState, self.Q, epsilon)
                obs, reward, done, _ = self.env.step(action)
                currentState = self.get_state(obs)

                #actualizo Q
                self.Q[previousState, action] = self.Q[previousState, action] + \
                    alpha * (reward + gamma * self.maxAction(currentState) - self.Q[previousState, action])
                #print('->', state, action, reward, obs, done)
            #guardo el Q optimo de un estado (inicial)
            initial_state_Q[count] = self.maxAction((0,0))
            count = count + 1
            #actualizar epsilon
        #Retornar el Q
        return self.Q
    
    def maxAction(self, state):
        return np.argmax(self.Q[state])
    
    #metodo de ejecucion
    def execute(self, iterations):
        count = 0
        execution_rewards = np.array(iterations)
        while count < iterations:
            reward_total = 0
            obs = self.env.reset()
            done = False
            while not done:
                action = self.maxAction(obs)
                obs, reward, done, _ = self.env.step(action)
                reward_total += reward
                self.env.render()
            
            execution_rewards[count] = reward_total
            count = count + 1
        self.rewards.append(statistics.mean(execution_rewards))

        #plot validacion con promedio de recompensas