import os
import Config
import numpy as np
import random
from PIL import Image
import cv2
import math
import matplotlib.pyplot as plt


def get_table(filename):
    '''
    Input : File
    Output : Array containing the table in the file in float format.
    '''
    
    f = open(filename, 'r')
    lines = f.readlines()
    l=[]
    for line in enumerate(lines):
        #print line
        line_split = line[1].split(' ')
        l.append(float(line_split[1]))
    return np.asarray(l)


def get_HW(filename):
    '''
    Output : Shape of the image.
    '''

    f = open(filename, 'r')
    lines = f.readlines()
    for i,line in enumerate(lines):
        if line.find('ROOM') > -1:
            return np.int(line.split(' ')[1]),np.int(line.split(' ')[2])
    return -1,-1



def extract_BB_coordinates(filename,camera):
    '''
    In : pom file name, camera id
    Out : List of all bounding boxes coordinates on this view, as defined by the pom file.
    '''
    
    f = open(filename, 'r')
    lines = f.readlines()
    bounding_boxes =[]
    current_object =1
    for i,line in enumerate(lines):

        if line.find('RECTANGLE %d'%camera) > -1:
            bounding_boxes.append(parse_BB_from_line(line))
    return bounding_boxes
            
def parse_BB_from_line(line):
    
    '''
    In : line string
    Out : coordinates of the box in the parsed line, where we set random 0-size coordinates.
    '''

    line_split = line.split(' ')
    if line_split[3] == 'notvisible\n':
        return [0,0,0,0]
    else:
        return [np.int(line_split[3]),np.int(line_split[4]),np.int(line_split[5]),np.int(line_split[6])]



