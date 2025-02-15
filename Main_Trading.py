import torch
import torch.optim as optim
import torch.nn.functional as F

import QValues_Trading
from Agent_Trading import AgentTrading
from DQN_Trading import DQN
from Environment_Trading import TradingEnvironment
from Experience import Experience, extract_tensors
from Plotter import plot
from ReplayMemory import ReplayMemory
from Strategy import EpsilonGreedyStrategy


batch_size = 32
gamma = 0.999
eps_start = 1
eps_end = 0.01
eps_decay = 0.0005
target_update = 200
memory_size = 50000
lr = 0.0001

print("\033[92m(Main) Using batch size of: " + str(batch_size))
print("(Main) Using learning rate of: " + str(lr))
print("(Main) Updating target nn for episodes: " + str(target_update))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
environment = TradingEnvironment(True)
strategy = EpsilonGreedyStrategy(eps_start, eps_end, eps_decay)
agent = AgentTrading(strategy, environment, device)
memory = ReplayMemory(memory_size)

policy_net = DQN(environment.get_windows_length()).to(device)
target_net = DQN(environment.get_windows_length()).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
optimizer = optim.Adam(params=policy_net.parameters(), lr=lr)


episode_rewards = []
rewards_average = []
episode_number = 0


def get_average(values, period):
    if len(values) >= period:
        average = sum(values[len(values) - period:]) / period
        return average
    else:
        return 0


input("\033[94mCheck the parameters above and enter any key to start the training: ")


# while environment.has_episodes():
for i in range(100000):
    episode_number += 1
    environment.reset()
    state = environment.get_state()
    episode_reward = 0
    while True:
        action = agent.select_action(state, policy_net)
        reward, done = environment.take_action(action)
        episode_reward += reward
        if not done:
            next_state = environment.get_state()
            state = next_state
        else:
            # If the episode was done, create a fake next state that indicate that is done
            next_state = (0.0, 0.0, -1.0, [0.0] * len(state[3]), [0.0] * len(state[4]))

        memory.push(Experience(DQN.convert_to_tensor(state, device), torch.tensor([action.value]).to(device),
                               DQN.convert_to_tensor(next_state, device), torch.tensor([reward]).to(device)))

        if memory.can_provide_sample(batch_size):
            experiences = memory.sample(batch_size)
            states, actions, rewards, next_states = extract_tensors(experiences)
            current_q_values = QValues_Trading.QValuesTrading.get_current(policy_net, states, actions)
            next_q_values = QValues_Trading.QValuesTrading.get_next(target_net, next_states)
            target_q_values = (next_q_values * gamma) + rewards

            loss = F.mse_loss(current_q_values, target_q_values.unsqueeze(1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if done:
            episode_rewards.append(reward)
            average = get_average(episode_rewards, 100)
            rewards_average.append(average)
            if average > 9 and batch_size != 256:
                print("Changing batch")
                batch_size = 256
            plot(episode_rewards, rewards_average)
            break

    if episode_number % target_update == 0:
        target_net.load_state_dict(policy_net.state_dict())


