import sys
import numpy as np
import matplotlib.pyplot as plt
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple
import seaborn as sns
from math import sqrt
from agent import Agent
import argparse

class Env:
    def __init__(self, name, agent, train = True):
        self.env = UnityEnvironment(file_name=name)
        self.env.reset()
        self.name = list(self.env.behavior_specs)[0]
        self.n_actions = self.env.behavior_specs[
            self.name
        ].action_spec.discrete_branches
        self.agent = agent
        self.episodes = []
        self.training = train

    def episode(self):
        self.env.reset()
        decision_steps, terminal_steps = self.env.get_steps(self.name)
        tracked_agent = -1  # -1 indicates not yet tracking

        def split(x):
            x += 4
            return x[0, (0, 2)], x[0, (3, 5)]

        obs = tuple(map(split, decision_steps.obs))
        n_steps = 0
        running = True
        while running:
            n_steps += 1
            if tracked_agent == -1 and len(decision_steps) >= 1:
                tracked_agent = decision_steps.agent_id[0]

            action = self.agent.get_action(obs[0][0], obs[0][1]) + 1
            self.env.set_actions(self.name, ActionTuple(discrete=np.array([[action]])))
            self.env.step()
            decision_steps, terminal_steps = self.env.get_steps(self.name)

            if tracked_agent in terminal_steps:
                obs = terminal_steps.obs
                reward = terminal_steps[tracked_agent].reward
                running = False

            elif tracked_agent in decision_steps:
                obs = decision_steps.obs
                reward = decision_steps[tracked_agent].reward

            obs = tuple(map(split, obs))

            if self.training:
                self.agent.update_weights(reward)

        return n_steps

    def train(self, n_episodes):
        for episode in range(n_episodes):

            if self.training:
                self.agent.update_params(episode)
                
            steps = self.episode()
            self.episodes.append(steps)
            print(f"Episode {episode+1}, Steps: ",steps )

    def plot_steps(self):
        sns.scatterplot(data=self.episodes)
        plt.show()

    def close(self):
        self.env.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps","-n",type=int, default=50, help="Number of steps to run")
    parser.add_argument("--load_weights", action="store_true", help="Use pretrained weights")
    args = parser.parse_args()
    agent = Agent()

    if args.load_weights:
        agent.load_weights("weights")

    env = Env("GridyWorld", agent, not args.load_weights)
    env.train(args.steps)
    env.plot_steps()

if __name__ == "__main__":
    main()