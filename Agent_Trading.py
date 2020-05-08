import random
import torch


class AgentTrading():
    def __init__(self, strategy, environment, device):
        self.current_step = 0
        self.strategy = strategy
        self.device = device
        self.environment = environment

    def select_action(self, state, policy_net):
        rate = self.strategy.get_exploration_rate(self.current_step)
        self.current_step += 1

        if rate > random.random():  # Explore
            action = random.choice(self.environment.get_possible_actions())
            return action
        else:
            with torch.no_grad():
                # TODO: get all the probablities for all the actions and pick the one with with the greatest posiblity that is included in the possible actions
                # TODO: how do you know which probability corresponds to a state?
                return policy_net(state)  # Exploit maximo


