from typing import Any, Dict
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

from rl_hpo.baes import AbsAgent
from rl_hpo.models.dqn import DQNModel
from rl_hpo.constants import STATE, ACTION, REWARD, NEXT_STATE, DONE

import random
from collections import deque
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

from rl_hpo.params.dqn import DQNHyperParamsWithTune


class DQNAgent(AbsAgent):
    def __init__(
        self,
        state_size: int,
        action_size: int,
        config: Dict[str, Any] 
    ):
        self.state_size = state_size
        self.action_size = action_size

        # Hyperparameters from config
        self.discount_factor = config.get('discount_factor', 0)
        self.learning_rate = config.get('learning_rate', 0)
        self.epsilon = config.get('epsilon', 0)
        self.epsilon_decay = config.get('epsilon_decay', 0)
        self.epsilon_min = config.get('epsilon_min', 0)
        self.batch_size = config.get('batch_size', 0)
        self.train_start = config.get('train_start', 0)
        self.target_update_period = config.get('target_update_period', 0)
        self.hidden_size = config.get('hidden_size', 0)

        self.step_counter = 0

        # Replay Buffer
        self.memory: deque[tuple] = deque(maxlen=10000)

        # Model and Target Model
        self.policy_net = DQNModel(state_size, action_size, self.hidden_size)
        self.target_model = DQNModel(state_size, action_size, self.hidden_size)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)

        # Initial Synchronization of Target Model
        self.update_target_model()

        # For logging
        self.loss_history: list[float] = []

    def get_action(self, state: np.ndarray) -> int:
        self.step_counter += 1
        if random.random() <= self.epsilon:
            return random.randrange(0, self.action_size)
        else:
            with torch.no_grad():
                return self.policy_net(
                    torch.FloatTensor(state)
                ).argmax().item()

    def train(self):
        if len(self.memory) < self.train_start:
            return

        mini_batch = random.sample(self.memory, self.batch_size)

        states = torch.FloatTensor(np.array([x[STATE] for x in mini_batch]))    # [batch_size, state_size]
        actions = torch.LongTensor(np.array([x[ACTION] for x in mini_batch]))   # [batch_size]
        rewards = torch.FloatTensor(np.array([x[REWARD] for x in mini_batch]))  # [batch_size]
        next_states = torch.FloatTensor(np.array([x[NEXT_STATE] for x in mini_batch]))  # [batch_size, state_size]
        dones = torch.FloatTensor(np.array([x[DONE] for x in mini_batch]))      # [batch_size]

        # Current Q Values: Q(s,a)
        curr_Q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)  # [batch_size]

        # Next Q Values: max_a' Q'(s',a')
        next_Q = self.target_model(next_states).max(1)[0]  # [batch_size]
        expected_Q = rewards + self.discount_factor * next_Q * (1 - dones)  # [batch_size]

        # Loss
        loss = F.mse_loss(curr_Q, expected_Q.detach())

        # Update Parameters
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Log Loss
        self.loss_history.append(loss.item())

        if self.step_counter % self.target_update_period == 0:
            self.update_target_model()

        # Decrease epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_model(self):
        self.target_model.load_state_dict(self.policy_net.state_dict())

    def done(self):
        self.update_target_model()

    def evaluate(self): ...

    def update(self):
        self.update_target_model()
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))