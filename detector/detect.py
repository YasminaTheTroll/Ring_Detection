import cv2
import numpy as np
import math
import matplotlib.pyplot as plt




def detect(img, low, high):
    
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV_FULL)
    mask = cv2.inRange(img_hsv, low, high)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

    matches = []
    
    for c in contours:
        perimeter = cv2.arcLength(c, True)
        if perimeter == 0:
            continue
            
        area = cv2.contourArea(c)
        roundness = 4 * math.pi * area/(perimeter * perimeter)
        
        full_match = roundness > 0.6
        x,y,w,h = cv2.boundingRect(c)

        if w > 0.06 * img.shape[1]  and h > 0.03 * img.shape[0]:
            match = Match(w, h, x, y, full_match)
            matches.append(match)

    return matches




class Match:
    def __init__(self, w, h, x, y, full = False):
        #x and y are topleft point coords
        self.w = w
        self.h = h
        self.x = x
        self.y = y
        self.full = full

        
    def center(self):
        h = self.x + (self.w / 2)
        k = self.y + (self.h / 2)
        return (h, k)
        
    def show(self, img):
        color = [0, 255, 255]
        if self.isFull():
            color = [63, 255, 0]

        return cv2.rectangle(img, (self.x, self.y), (self.x + self.w, self.y + self.h), color, 3)
            
                     
    def isFull(self):
        return self.full

    def isPartial(self):
        return not self.full