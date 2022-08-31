from tokenize import StopTokenizing
from numpy.lib.npyio import load
from model import DinoNet
import torch
from collections import deque
import numpy as np
import random
from pathlib import Path
import time

class Dino:
    def __init__(self,
        state_dim, 
        action_dim, 
        mem_size, 
        batch_size, 
        save_dir,
        learning_rate = 0.00025,
        checkpoint=None):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.mem = deque(maxlen=mem_size)
        self.batch_size = batch_size
        
        self.epsilon = 1
        self.epsilon_decay = 0.99999975
        self.epsilon_rate_min = 0.1
        self.gamma = 0.9

        self.current_step = 0
        #self.burnin = 1e5
        self.burnin = 100
        self.learn_every = 3
        self.sync = 1e4

        self.save = 5e5
        self.save_dir = save_dir

        self.use_cuda = torch.cuda.is_available()

        self.net = DinoNet(input_dim=self.state_dim, output_dim=action_dim).float()
        if self.use_cuda:
            self.net = self.net.to(device='cuda')
        if checkpoint:
            self.load(checkpoint)

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=learning_rate)
        self.loss_fn = torch.nn.SmoothL1Loss()

    def act(self, state):
        if np.random.rand() < self.epsilon:
            action_ = np.random.randint(self.action_dim)
        else:
            state = torch.FloatTensor(state).cuda() if self.use_cuda else torch.FloatTensor(state)
            state = state.unsqueeze(0)
            action_values = self.net(state, model='qnet')
            action_ = torch.argmax(action_values, axis=1).item()
        
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_rate_min, self.epsilon)

        self.current_step += 1
        return action_
    
    def cache(self, state, next_state, action, reward, done):
        state = torch.FloatTensor(state).cuda() if self.use_cuda else torch.FloatTensor(state)
        next_state = torch.FloatTensor(next_state).cuda() if self.use_cuda else torch.FloatTensor(next_state)
        action = torch.LongTensor([action]).cuda() if self.use_cuda else torch.LongTensor([action])
        reward = torch.DoubleTensor([reward]).cuda() if self.use_cuda else torch.DoubleTensor([reward])
        done = torch.BoolTensor([done]).cuda() if self.use_cuda else torch.BoolTensor([done])

        self.mem.append((state, next_state, action, reward, done))
    
    def get_batch(self):
        batch = random.sample(self.mem, self.batch_size)
        state, next_state, action, reward, done = map(torch.stack, zip(*batch))
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()

    def estimate_q(self, state, action):
        current_q = self.net(state, model='qnet')[np.arange(0, self.batch_size), action]
        return current_q
    
    @torch.no_grad()
    def target_q(self, reward, next_state, done):
        next_state_q = self.net(next_state, model='qnet')
        best_action = torch.argmax(next_state_q, axis=1)
        next_q = self.net(next_state, model='qhatnet')[np.arange(0, self.batch_size), best_action]
        return (reward + (1 - done.float()) * self.gamma * next_q).float()
    
    def update_q_net(self, estimate_q, target_q):
        loss = self.loss_fn(estimate_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def sync_q_nets(self):
        self.net.qhatnet.load_state_dict(self.net.qnet.state_dict())
    
    def learn(self):
        start = time.time()
        if self.current_step % self.sync == 0:
            self.sync_q_nets()
        
        if self.current_step % self.save == 0:
            self.save()
        
        if self.current_step < self.burnin:
            print(self.current_step)
            return None, None
        
        if self.current_step % self.learn_every != 0:
            return None, None
        
        state, next_state, action, reward, done = self.get_batch()

        q_est = self.estimate_q(state, action)

        q_target = self.target_q(reward, next_state, done)

        loss = self.update_q_net(q_est, q_target)
        end = time.time()
        if(end-start < 0.1):#small sleep to not crash the game
            time.sleep(0.1)
        
        return (q_est.mean().item(), loss)
    
    def save_model(self):
        path = self.save_dir / f"dino_net_{int(self.current_step // self.save)}.chkpt"
        torch.save(
            dict(
                model = self.net.state_dict(),
                epsilon = self.epsilon
            ),
            path
        )
        print(f"Dino net save | step: {self.current_step}")
    
    def load(self, path):
        if not path.exists():
            raise ValueError(f"{load} does not exist")
        
        chkpt = torch.load(path, map_location=('cuda' if self.use_cuda else 'cpu'))
        self.epsilon = chkpt.get('epsilon')
        state_dict = chkpt.get('model')
        self.net.load_state_dict(state_dict)

        print(f"Loading model | epsilon: {self.epsilon}")
