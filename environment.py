from ast import get_source_segment
import cv2
import numpy as np
from mss import mss
from PIL import Image
from numpy.core.fromnumeric import size
from selenium import webdriver
from selenium.webdriver.common import keys
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import WebDriverException
from torch.random import get_rng_state
import time
from webdriver_manager.chrome import ChromeDriverManager

class DinoGameEnv():
    
    def __init__(self,
        size_screen, 
        size_resize):
        self.driver = webdriver.Chrome(ChromeDriverManager().install())
        self.driver.set_window_size(500,500)
        self.driver.set_window_position(-14,-14)
        
        try:
            self.driver.get('chrome://dino/')
        except WebDriverException:
            pass
        self.button = self.driver.find_element(By.TAG_NAME, "body")
        self.size_screen = size_screen
        self.size_resize = size_resize

    def get_screen_image(self):
        sct = mss()
        img = Image.frombytes('RGB', (self.size_screen['width'],self.size_screen['height']), sct.grab(self.size_screen).rgb)
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
        img = cv2.resize(img, (self.size_resize['width'],self.size_resize['height']))
        #cv2.imwrite('/home/gilson/Documents/RL/chrome-dino-reinforcement-learning/img.png',img)
        return img

    def get_state(self):
        frame1 = self.get_screen_image()
        frame2 = self.get_screen_image()
        frame3 = self.get_screen_image()
        frame4 = self.get_screen_image()
        return np.dstack((frame1, frame2, frame3, frame4)).transpose(2,1,0)

    def execute_action(self, action):
        #if(action == 'none'):
        #    next_state = self.get_state()
        #    reward = 0
        #    done = self.get_status()
        #    return next_state, reward, done 
        if(action == 'up'):
            self.button.send_keys(Keys.ARROW_UP)
        elif(action == 'down'):
            webdriver.ActionChains(self.driver).key_down(Keys.ARROW_DOWN).perform()
            webdriver.ActionChains(self.driver).key_up(Keys.ARROW_DOWN).perform()
        elif(action == 'hold_down'):
            webdriver.ActionChains(self.driver).key_down(Keys.ARROW_DOWN).perform()
        elif(action == 'release_down'):
            webdriver.ActionChains(self.driver).key_up(Keys.ARROW_DOWN).perform()
        
        next_state = self.get_state()
        #reward = self.get_score()
        reward = 1
        done = self.get_status()
        
        if(done):
            reward = -1

        return next_state, reward, done 

    def pause_game(self, b):
        if(b):
            self.driver.execute_script("Runner.instance_.onVisibilityChange(new Event('blur'))")
        else:
            self.driver.execute_script("Runner.instance_.onVisibilityChange(new Event('click'))")
    def get_score(self):
        return self.driver.execute_script("return parseInt(String(Runner.instance_.distanceMeter.digits).replaceAll(',', ''))")

    def start_game(self):
        if(self.get_status()):
            time.sleep(1.5)
            self.button.send_keys(Keys.SPACE)
        else:
            activated = self.driver.execute_script("return Runner.instance_.activated")
            if not activated:
                self.button.send_keys(Keys.SPACE)
    
    def get_status(self):
        return self.driver.execute_script("return Runner.instance_.crashed")

    def __del__(self):
        self.driver.close()
    
