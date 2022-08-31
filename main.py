from datetime import datetime
import cv2
import time
from torch.cuda import _sleep
from tqdm import tqdm
import numpy as np
from mss import mss
from PIL import Image
from selenium import webdriver
from selenium.common import exceptions
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import WebDriverException
from selenium.webdriver.common.action_chains import ActionChains

from environment import DinoGameEnv
from agent import Dino
from logger import Logger

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, help='mode of execution(train | eval)')
parser.add_argument('--eps', type=int, help='number of episodes')
args = parser.parse_args()

#width of game screen
w,h = 500,200
size = {'top':200, 'left':2, 'width':w, 'height':h}

action_set = {
    0: 'up',
    1: 'down',
    2: 'hold_down',
    3: 'release_down',
    4: 'none'
}


DinoGameEnv_ = DinoGameEnv(size, {'width': 140, 'height':140})
DinoAgent = Dino((4, 140, 140), 3, 10000, 32, 'learn_weights/' + datetime.now().strftime('%d-%m-%Y%H-%M-%S'))
logger = Logger('logs/')

if args.mode == 'train':
    for i in tqdm(range(args.eps)):
        DinoGameEnv_.start_game()
        state = DinoGameEnv_.get_state()
        print(state.shape)
        time.sleep(0.5)
        while 1:
            action = DinoAgent.act(state)

            next_state, reward, done = DinoGameEnv_.execute_action(action_set[action])

            DinoAgent.cache(state, next_state, action, reward, done)
            
            DinoGameEnv_.pause_game(True)
            q, loss = DinoAgent.learn()
            DinoGameEnv_.pause_game(False)
            
            state = next_state
            
            print('reward: ' ,reward)
            print('loss: ' ,loss)
            print('q: ' , q)
            
            logger.log_step(reward, loss, q)

            if done:
                break
        
        logger.log_episode()

        if i % 20 == 0:
            logger.record(
                episode=i,
                epsilon=DinoAgent.epsilon,
                step=DinoAgent.current_step
            )
