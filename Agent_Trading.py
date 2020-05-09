import random
import torch

from DQN_Trading import DQN
from Environment_Trading import Actions


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
        else:  # Exploit
            with torch.no_grad():
                # Get the actions that can be taken in the current state
                possible_actions = self.environment.get_possible_actions()
                # Get the Q values for all the actions using the DQN
                values = policy_net(DQN.convert_to_tensor(state)).numpy()[0]
                # Store the Q values with their corresponding actions in a dictionary
                action_values = [
                    {'action': Actions.WAIT, 'Qvalue': values[0]},
                    {'action': Actions.BUY, 'Qvalue': values[1]},
                    {'action': Actions.SELL, 'Qvalue': values[2]}
                ]
                # Sort the dictionary in descending order of Qvalues
                action_values.sort(key=lambda x: x['Qvalue'], reverse=True)
                # Pop the action with the greatest !value until find one that is possible
                best_action = action_values.pop(0)
                while best_action not in possible_actions:
                    if len(possible_actions) == 0:
                        raise ValueError("No possible actions to be taken")
                    best_action = action_values.pop(0)
                # Return the best action possible
                return best_action


