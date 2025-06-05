import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
import torchvision.transforms as T
import matplotlib.pyplot as plt

from CurriculumGym import CurriculumGym

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# class ImageEnvWrapper(gym.ObservationWrapper):
#     def __init__(self, env, num_stack=4):
#         super().__init__(env)
#         self.num_stack = num_stack
#         self.frames = deque([], maxlen=num_stack)
#
#         # Update observation space
#         self.observation_space = gym.spaces.Box(
#             low=0, high=255, shape=(num_stack, 84, 84), dtype=np.uint8
#         )
#
#         self.transform = T.Compose([
#             T.ToPILImage(),
#             T.Grayscale(),
#             T.Resize((84, 84)),
#             T.ToTensor()  # output shape: [1, 84, 84], values in [0, 1]
#         ])
#
#     def observation(self, obs):
#         frame = self.render()                     # Get RGB frame from env
#         frame = self.transform(frame).squeeze(0)  # Grayscale: [84, 84]
#         frame = (frame * 255).byte()              # Convert to uint8
#
#         self.frames.append(frame)
#
#         while len(self.frames) < self.num_stack:
#             self.frames.append(frame.clone())  # pad with initial frame
#
#         stacked = torch.stack(list(self.frames), dim=0)  # [4, 84, 84]
#         return stacked.numpy()
#
#     def reset(self, **kwargs):
#         obs, info = self.env.reset(**kwargs)
#         self.frames.clear()
#         return self.observation(obs), info

# --- Replay Buffer ---
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, transitions):
        for transition in transitions:
            self.buffer.append((transition.state, transition.action, transition.reward, transition.next_state, transition.done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)

        # Replace None next_states with zero tensors
        next_state = [
            torch.zeros_like(state[0]) if s is None else s
            for s in next_state
        ]

        return (
            torch.stack(state),
            torch.tensor(action),
            torch.tensor(reward, dtype=torch.float),
            torch.stack(next_state),
            torch.tensor(done, dtype=torch.float)
        )

    def __len__(self):
        return len(self.buffer)

# --- CNN-based Q-Network ---
class DQN(nn.Module):
    def __init__(self, input_h, input_w, num_actions):
        super().__init__()
        self.num_actions = num_actions
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        # Dynamically compute flattened size
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, input_h, input_w)
            dummy_output = self.conv(dummy_input)
            self.flattened_size = dummy_output.view(1, -1).size(1)

        self.fc = nn.Linear(self.flattened_size, 512)

        self.value_stream = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))

        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        return value + (advantage - advantage.mean(dim=1, keepdim=True))

    def select_actions(self, states, eps, test_mode):
        actions = []
        with torch.no_grad():
            q_vals = self.forward(states)  # states shape: [batch, C, H, W]
            greedy_actions = q_vals.argmax(dim=1)  # shape: [batch]

        for i in range(len(states)):
            if random.random() < eps and test_mode == False:
                actions.append(random.randint(0, self.num_actions - 1))  # random action
            else:
                actions.append(greedy_actions[i].item())  # greedy action

        return torch.tensor(actions)


# --- Hyperparameters ---
env = CurriculumGym()
num_actions = env.n_actions
q_net = DQN(env.input_h, env.input_w, num_actions).to(device)
target_net = DQN(env.input_h, env.input_w, num_actions).to(device)
target_net.load_state_dict(q_net.state_dict())

optimizer = torch.optim.Adam(q_net.parameters(), lr=1e-4)
buffer = ReplayBuffer(10000)
batch_size = 32
gamma = 1
epsilon = 1
eps_decay = 0.1
eps_min = 0.05
n_updates = 0
target_update_freq = 10
evaluate_freq = 25
episodes = 2500
epoch = 25
best = -1e5
test_freq = 25

# --- Training Loop ---
returns = []
for episode in range(episodes):
    env.reset()
    transitions = env.rollout(q_net, epsilon)
    buffer.push(transitions.values())
    total_reward = 0
    for transition in transitions.values():
        total_reward += transition.reward

    if len(buffer) >= batch_size:
        for _ in range(epoch):
            states, actions, rewards, next_states, dones = buffer.sample(batch_size)
            states = states.to(device)
            actions = actions.to(device)
            rewards = rewards.to(device)
            next_states = next_states.to(device)
            dones = dones.to(device)

            q_vals = q_net(states).gather(1, actions.unsqueeze(1)).squeeze()
            with torch.no_grad():
                next_actions = q_net(next_states).argmax(dim=1, keepdim=True)
                max_next_q_vals = target_net(next_states).gather(1, next_actions).squeeze()
                target = rewards + gamma * max_next_q_vals * (1 - dones)

            loss = F.mse_loss(q_vals, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            n_updates += 1

    if n_updates % target_update_freq == 0:
        target_net.load_state_dict(q_net.state_dict())

    if n_updates % evaluate_freq == 0:
        test_total_reward = 0
        for test_tran in env.rollout(q_net, epsilon, True).values():
            test_total_reward += test_tran.reward
        if test_total_reward > best:
            best = test_total_reward
            torch.save(q_net.state_dict(), 'best_dqn_weights.pth')

    epsilon = max(epsilon * eps_decay, eps_min)
    returns.append(total_reward)
    print(f"Episode {episode}: Return = {total_reward}")
    with open('train_result.txt', 'w') as f:
        for ret in returns:
            f.write(f"{ret}\n")

env.close()
plt.plot(returns)
plt.xlabel("Episode")
plt.ylabel("Return")
plt.title("DQN with Image Observations")
plt.show()
