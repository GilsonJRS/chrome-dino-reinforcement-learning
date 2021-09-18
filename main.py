import cv2
import time
import numpy as np
from mss import mss
from PIL import Image
from selenium import webdriver
from selenium.common import exceptions
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import WebDriverException
from selenium.webdriver.common.action_chains import ActionChains


#width of game screen
w,h = 500,200
size = {'top':200, 'left':4, 'width':w, 'height':h}
