import math

import torch


def adapt_qvalues(buy_state, state_hour, qvalues, i):
    if buy_state == 0:
        if state_hour == 23:
            qvalues[i][2] = - math.inf
            qvalues[i][1] = - math.inf
        else:
            qvalues[i][2] = -math.inf
    else:
        if state_hour == 23:
            qvalues[i][0] = - math.inf
            qvalues[i][1] = - math.inf
        else:
            qvalues[i][1] = -math.inf


class QValuesTrading():
    device = torch.device("cpu")

    @staticmethod
    def get_current(policy_net, states, actions):
        return policy_net(states).gather(dim=1, index=actions.unsqueeze(-1))

    @staticmethod
    def get_next(target_net, next_states):
        qvalues = target_net(next_states)
        for i in range(len(next_states)):
            state = next_states[i]
            buy_state = state[0][1]
            state_hour = state[1][-1]
            adapt_qvalues(buy_state, state_hour, qvalues, i)
        return qvalues.max(dim=1)[0].detach() #.view(len(next_states), -1)

