import numpy as np
from MountainCarEnv import MountainCarEnv
import statistics


class QLearning():

    def __init__(self, render):
        self.env = MountainCarEnv(render_mode = render)
        self.actions = list(range(self.env.action_space.n))
        self.rewards = []
        self.initial_state = (0,0)
            
    def discretizar(self, divPos, divVel):
        self.pos_space = np.linspace(-1.1, 0.5, divPos)
        self.vel_space = np.linspace(-0.07, 0.07, divVel)
        self.Q = np.zeros((divPos+1, divVel+1, 3))

    def get_state(self, obs):
        pos, vel = obs
        pos_bin = np.digitize(pos, self.pos_space)
        vel_bin = np.digitize(vel, self.vel_space)
        return pos_bin, vel_bin

    def epsilon_greedy_policy(self, state, Q, epsilon=0.1):
        explore = np.random.binomial(1, epsilon)
        #epsilon alto es mucho explore
        if explore:
            action = self.env.action_space.sample()
        # exploit
        else:
            action = np.argmax(Q[state])
        return action

    def optimal_policy(state, Q):
        action = np.argmax(Q[state])
        return action

    def qLearning(self, iterations, alpha, epsilon, gamma):
        count = 0
        initial_state_Q = []
        while(count < iterations):
            obs = self.env.reset()
            # initial_state = self.get_state(obs)
            done = False
            while not done:
                previousState = self.get_state(obs)
                action = self.epsilon_greedy_policy(previousState, self.Q, epsilon)
                obs, reward, done, _ = self.env.step(action)
                currentState = self.get_state(obs)

                #actualizo Q
                self.Q[previousState][action] = self.Q[previousState][action] + \
                    alpha * (reward + (gamma * self.maxQ(currentState)) - self.Q[previousState][action])
                # -500 + (-1 -499 + 500)
            #guardo el Q optimo de un estado (inicial)
            initial_state_Q.append(self.maxQ(self.initial_state))

            #actualizar epsilon
                #esto no va para este test

            count = count + 1
        #Retornar el Q
        ## --> guardar pkl
        return self.Q, initial_state_Q
    
    def maxQ(self, state):
        action = np.argmax(self.Q[state])
        return self.Q[state][action]

    def maxAction(self, state):
        return np.argmax(self.Q[state])
    
    #metodo de ejecucion
    def execute(self, iterations):
        count = 0
        execution_rewards = []
        while count < iterations:
            reward_total = 0
            obs = self.env.reset()
            done = False
            while not done:
                state = self.get_state(obs)
                action = self.maxAction(state)
                obs, reward, done, _ = self.env.step(action)
                reward_total += reward
                self.env.render()
            
            execution_rewards.append(reward_total)
            count = count + 1
        self.rewards.append(statistics.mean(execution_rewards))
        return statistics.mean(execution_rewards)

        #plot validacion con promedio de recompensas
    
    def setQ(self, Q):
        self.Q = Q

    def qLearning_bajar_epsilon_1(self, max_iterations, alpha, initial_epsilon, gamma):
        count = 0
        initial_state_Q = []
        epsilon_decay_list = []
        final_epsilon = 0.01
        while(count < max_iterations):
            epsilonDecreaseRate = max(((max_iterations - count) / max_iterations), 0)
            epsilon = ((initial_epsilon - final_epsilon) * epsilonDecreaseRate) + final_epsilon
            epsilon_decay_list.append(epsilon)
            obs = self.env.reset()
            done = False
            while not done:
                previousState = self.get_state(obs)
                action = self.epsilon_greedy_policy(previousState, self.Q, epsilon)
                obs, reward, done, _ = self.env.step(action)
                currentState = self.get_state(obs)

                self.Q[previousState][action] = self.Q[previousState][action] + \
                    alpha * (reward + (gamma * self.maxQ(currentState)) - self.Q[previousState][action])
                
            initial_state_Q.append(self.maxQ(self.initial_state))

            count = count + 1
        return self.Q, initial_state_Q, epsilon_decay_list
    

    #Q learning de mateo que supuestamente funciona
    def qLearning_2(self, episodes, 
                    learning_rate, 
                    discount_factor, 
                    exploration_rate, 
                    max_exploration_rate, 
                    min_exploration_rate,
                    exploration_decay_rate):
        initial_state_Q = []
        for episode in range(episodes):
            obs = self.env.reset()
            done = False
            total_reward = 0

            while(not done):
                state = self.get_state(obs)
                # Elige una acción utilizando la política epsilon-greedy
                exploration_rate_threshold = np.random.uniform(0, 1)
                if exploration_rate_threshold > exploration_rate:
                    action = np.argmax(self.Q[state[0],state[1]])
                else:
                    action = self.env.action_space.sample()

                # Realiza la acción y observa el nuevo estado y la recompensa
                new_obs, reward, done, _ = self.env.step(action)
                new_state = self.get_state(new_obs)
                # Actualiza los valores de Q utilizando la ecuación de Q-Learning
                self.Q[state[0],state[1],action] += learning_rate * (
                    reward + discount_factor * np.max(self.Q[new_state[0],new_state[1]]) - self.Q[state[0],state[1],action] 
                )

                total_reward += reward
                obs = new_obs
            initial_state_Q.append(self.maxQ(self.initial_state))
        