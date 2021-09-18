import cv2
import numpy as np
from mss import mss
from PIL import Image
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import WebDriverException

class DinoGameEnv():
    
    def __init__(self):
        self.driver = webdriver.Chrome('/home/gilson/Downloads/chromedriver')
        self.driver.set_window_position(0,0)
        self.driver.set_window_size(500,500)
        try:
            self.driver.get('chrome://dino/')
        except WebDriverException:
            pass
        self.button = self.driver.find_element(By.TAG_NAME, "body")
    
    def getScreenImage(size):
        sct = mss()
        img = Image.frombytes('RGB', (size['width'],size['height']), sct.grab(size).rgb)
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
        img = cv2.resize(img, (250,100))
        return img
    
    def executeAction(self, action):
        if(action == 'up'):
            self.button.send_keys(Keys.SPACE)
        elif(action == 'down'):
            self.button.send_keys(Keys.DOWN)
    
    def getScore(self):
        return self.driver.execute_script("return parseInt(String(Runner.instance_.distanceMeter.digits).replaceAll(',', ''))")
    
    def __del__(self):
        self.driver.close()