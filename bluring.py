import cv2
import numpy as np
import sys
from matplotlib import pyplot as plt
import time
import os.path

read_directory = "Query_images"
write_directory = "Query_images"
sufex = "_blured_resize" #no sufex means overwrite
filter_size = 5
number_of_passes_start = 1
number_of_passes_end = 1
fixed_size = True # if true then fixed size scaling else use down_scale
fix_x = 1280
fix_y = 720
down_scale = 0.9
number_of_downscales = 1
for file in os.listdir(read_directory):#os.getcwd()
    if file.endswith(".jpg"):
        # extraction of file name without extention to pares to an int
        name_without_extention = file.rsplit('.', 1)[0]
        img_template = cv2.imread(read_directory+str("/")+file)
        for iter in xrange(0,number_of_passes_start):
            img_template = cv2.GaussianBlur(img_template, (filter_size, filter_size), 0)
        if fixed_size:
            img_template = cv2.resize(img_template, (fix_x,fix_y))
        else:
            for iter in xrange(0,number_of_downscales):
                img_template = cv2.resize(img_template, (img_template.cols*down_scale,img_template.rows*down_scale))
                
        for iter in xrange(0,number_of_passes_end):
            img_template = cv2.GaussianBlur(img_template, (filter_size, filter_size), 0)
                
        cv2.imwrite(write_directory+str("/")+name_without_extention+sufex+'.jpg',img_template)