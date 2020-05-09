import torch
import torch.optim as optim
import torch.nn.functional as F

import QValues_Trading
from Agent_Trading import AgentTrading
from DQN_Trading import DQN
from Environment_Trading import TradingEnvironment
from Experience import Experience, extract_tensors
from ReplayMemory import ReplayMemory
from Strategy import EpsilonGreedyStrategy

batch_size = 256
gamma = 0.999
eps_start = 1
eps_end = 0.01
eps_decay = 0.001
target_update = 10
memory_size = 50000
lr = 0.001
num_episodes = 1000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
environment = TradingEnvironment()
strategy = EpsilonGreedyStrategy(eps_start, eps_end, eps_decay)
agent = AgentTrading(strategy, environment, device)
memory = ReplayMemory(memory_size)
# TODO: Code the DQN
policy_net = DQN(environment.get_screen_height(), environment.get_screen_width()).to(device)
target_net = DQN(environment.get_screen_height(), environment.get_screen_width()).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
optimizer = optim.Adam(params=policy_net.parameters(), lr=lr)

episode_rewards = []
episode_number = 0
while environment.has_episodes():
    episode_number += 1
    environment.reset()
    state = environment.get_state()
    episode_reward = 0
    while True:
        # TODO: check return type matches
        action = agent.select_action(state, policy_net)
        reward, done = environment.take_action(action)
        episode_reward += reward
        next_state = environment.get_state()
        memory.push(Experience(state, action, next_state, reward))
        state = next_state

        if memory.can_provide_sample(batch_size):
            experiences = memory.sample(batch_size)
            states, actions, rewards, next_states = extract_tensors(experiences)
            # TODO: Check QValues methods
            current_q_values = QValues_Trading.QValuesTrading.get_current(policy_net, states, actions)
            next_q_values = QValues_Trading.QValuesTrading.get_next(target_net, next_states)
            target_q_values = (next_q_values * gamma) + rewards

            loss = F.mse_loss(current_q_values, target_q_values.unsqueeze(1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if done:
            episode_rewards.append(reward)
            # TODO: plot rewards
            break

    if episode_number % target_update == 0:
        target_net.load_state_dict(policy_net.state_dict())

