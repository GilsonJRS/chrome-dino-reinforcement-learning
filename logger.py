import numpy as np
import time
from datetime import datetime
import matplotlib.pyplot as plt

class Logger():
    def __init__(self, path):
        
        #metrics
        self.eps_rewards = []
        self.eps_lengths = []
        self.eps_avg_losses = []
        self.eps_avg_qs = []

        #moving average
        self.moving_avg_eps_rewards = []
        self.moving_avg_eps_lengths = []
        self.moving_avg_eps_losses = []
        self.moving_avg_eps_qs = []

        #reset to new episode
        self.init_eps()

        #time
        self.record_time = time.time()
    
    def log_step(self, reward, loss, q):
        self.curr_eps_reward += reward
        self.curr_eps_length += 1
        if loss:
            self.curr_eps_loss += loss
            self.curr_eps_q += q
            self.curr_eps_loss_length += 1 

    def log_episode(self):
        self.eps_rewards.append(self.curr_eps_reward)
        self.eps_lengths.append(self.curr_eps_length)
        if self.curr_eps_loss_length == 0:
            eps_avg_loss = 0
            eps_avg_q = 0
        else:
            eps_avg_loss = np.round(self.curr_eps_loss / self.curr_eps_loss_length)
            eps_avg_q = np.round(self.curr_eps_q / self.curr_eps_loss_length, 5)
        self.eps_avg_losses.append(eps_avg_loss)
        self.eps_avg_qs.append(eps_avg_q)

        self.init_eps()

    def init_eps(self):
        self.curr_eps_reward = 0.0
        self.curr_eps_length = 0
        self.curr_eps_loss = 0.0
        self.curr_eps_q = 0.0
        self.curr_eps_loss_length = 0

    def record(self, episode, epsilon, step):
        mean_eps_reward = np.round(np.mean(self.eps_rewards[-100:]), 5)
        mean_eps_length = np.round(np.mean(self.eps_lengths[-100:]), 5)
        mean_eps_loss = np.round(np.mean(self.eps_avg_losses[-100:]), 5)
        mean_eps_q = np.round(np.mean(self.eps_avg_qs[-100:]), 5)
        self.moving_avg_eps_rewards.append(mean_eps_reward)
        self.moving_avg_eps_lengths.append(mean_eps_length)
        self.moving_avg_eps_losses.append(mean_eps_loss)
        self.moving_avg_eps_qs.append(mean_eps_q)

        print(
            f"Episode {episode} - "
            f"Step {step} - "
            f"Epsilon {epsilon} - "
            f"Mean Reward {mean_eps_reward} - "
            f"Mean Length {mean_eps_length} - "
            f"Mean Loss {mean_eps_loss} - "
            f"Mean Q Value {mean_eps_q} - "
            f"Time {datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}"
        )

        
