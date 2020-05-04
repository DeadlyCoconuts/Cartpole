import gym
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
import random
from PIL import Image
from model import DQN
from util import ReplayMemory
from util import Transition

import torch
import torch.optim as optim
import torchvision.transforms as T
import torch.nn.functional as F


BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10

resize = T.Compose([T.ToPILImage(), T.Resize(40, interpolation=Image.CUBIC), T.ToTensor()])

def get_cart_location(screen_width):
    world_width = env.x_threshold
    scale = screen_width / world_width
    return int(env.state[0] * scale + screen_width /2)  # middle of cart (screen starts from leftmost edge)

def get_screen():
    # Gym returns screen size of 800x1200x3 HWC. Transpose to Torch order (CHW).
    screen = env.render(mode='rgb_array').transpose((2, 0, 1))  # why permute and unpermute?
    _, screen_height, screen_width = screen.shape
    screen = screen[:, int(screen_height*0.4):int(screen_height*0.8), :]
    view_width = int(screen_width * 0.6)
    cart_location = get_cart_location(screen_width)
    if cart_location < view_width // 2:  # too far left
        slice_range = slice(view_width)  # end at view_width
    elif cart_location > (screen_width - view_width // 2):  # too far right
        slice_range = slice(-view_width, None)  # start at view_width
    else:  # centered
        slice_range = slice(cart_location - view_width // 2,
                            cart_location + view_width // 2)
    screen = screen[:, :, slice_range]
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255  # get contiguous array normalised
    screen = torch.from_numpy(screen)
    # Resize and add a batch dimension via unsqueeze
    return resize(screen).unsqueeze(0).to(device)

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1 * steps_done / EPS_DECAY)  # exp decay
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # accesses max out of batch > gets indices with [1] tuple indexing > resize indices tensor
            # TO TEST EFFECTS OF VIEW (1, 1)
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)

def plot_durations():
    plt.figure(2)  # creates another plot with id 2
    plt.clf()  # clears current figure
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plots them
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)  # gets moving set of 100 episodes > takes mean >
        # flattens
        means = torch.cat((torch.zeros(99), means))  # adds 99 0s as filler for the start
        plt.plot(means.numpy())

    plt.pause(0.001)  # give some time for plots to be updated

    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())

def optimize_model():
    if len(memory) < BATCH_SIZE:  # insufficient memory to optimize
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))  # Repack batch-array of Transitions to Transition of batch-arrays
    # Check this out

    # Final state will be marked with False
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)

    # Split batches up
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a)
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)  # set 0 everywhere so terminal states get 0
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()

    # Bootstrapping
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()  # update model


env = gym.make('CartPole-v0').unwrapped

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env.reset()
"""
plt.figure()
plt.imshow(get_screen().cpu().squeeze(0).permute(1, 2, 0).numpy(), interpolation='none')
plt.title('Example extracted screen')
plt.show()
"""

init_screen = get_screen()
_, _, screen_height, screen_width = init_screen.shape

n_actions = env.action_space.n

policy_net = DQN(screen_height, screen_width, n_actions).to(device)
target_net = DQN(screen_height, screen_width, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())  # loads weights
target_net.eval()  # sets target_net to eval mode; no training?

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(10000)  # create replay memory of size 10000

steps_done = 0

episode_durations = []

num_episodes = 300
for i_episode in range(num_episodes):
    # Initialize the environment and state
    env.reset()
    last_screen = get_screen()
    current_screen = get_screen()
    state = current_screen - last_screen
    for t in count():
        # Select and perform an action
        action = select_action(state)
        _, reward, done, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)

        # Observe new state
        last_screen = current_screen
        current_screen = get_screen()
        if not done:
            next_state = current_screen - last_screen
        else:
            next_state = None

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the target network)
        optimize_model()
        if done:
            episode_durations.append(t + 1)
            plot_durations()
            break
    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:  # update target network every 10 episodes
        target_net.load_state_dict(policy_net.state_dict())

print('Complete')
env.render()
env.close()
plt.ioff()
plt.show()










