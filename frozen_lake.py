# adapted from https://github.com/hiive/hiivemdptoolbox/blob/master/hiive/mdptoolbox/openai.py

# -*- coding: utf-8 -*-
import gym
import re
import numpy as np
from gym.envs.toy_text.frozen_lake import generate_random_map

def get_frozen_lake(n=8, frozen_p=0.7):
    map = generate_random_map(size=n, p=frozen_p)

    lake = OpenAI_FrozenLakeConverter(map, is_slippery=True, desc=map)

    return lake.P, lake.R, map



class OpenAI_FrozenLakeConverter:
    def __init__(self, random_map, **kwargs):
        """Create a new instance of the OpenAI_MDPToolbox class
        :param openAI_env_name: Valid name of an Open AI Gym env 
        :type openAI_env_name: str
        :param render: whether to render the Open AI gym env
        :type rander: boolean 
        """
        self.env_name = 'FrozenLake-v1'
    
        self.env = gym.make(self.env_name, **kwargs)
        self.env.reset()

        self.transitions = self.env.P
        self.actions = int(re.findall(r'\d+', str(self.env.action_space))[0])
        self.states = int(re.findall(r'\d+', str(self.env.observation_space))[0])
        self.P = np.zeros((self.actions, self.states, self.states))
        self.R = np.zeros((self.states, self.actions))
        self.convert_PR(random_map=random_map)
        
    def convert_PR(self, random_map=None):
        """Converts the transition probabilities provided by env.P to MDPToolbox-compatible P and R arrays
        """
        discretized_map = None
        if random_map is not None:
            discretized_map = "".join(random_map)
        for state in range(self.states):
            for action in range(self.actions):
                for i in range(len(self.transitions[state][action])):
                    tran_prob = self.transitions[state][action][i][0]
                    state_ = self.transitions[state][action][i][1]

                    if discretized_map is not None and discretized_map[state_] == "H" and discretized_map[state] != "H":
                        self.R[state][action] += tran_prob*-10
                    elif discretized_map is not None and discretized_map[state_] in ["F","S"]:
                        self.R[state][action] += tran_prob*-0.01
                    elif discretized_map is not None and discretized_map[state_] == "G" and discretized_map[state] != "G":
                        self.R[state][action] += tran_prob*10
                    else:
                        self.R[state][action] += tran_prob*self.transitions[state][action][i][2]
                    self.P[action, state, state_] += tran_prob