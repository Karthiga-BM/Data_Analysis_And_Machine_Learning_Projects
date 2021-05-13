import pandas as pd
import numpy as np
import cv2 #for image processing
import sys
import easygui
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk , Image
import imageio # to read image stored at particular path
import os

# Building a filebox to choose a particular file
def uploadfile():
    ImagePath =  easygui.fileopenbox() #This fileopenbox method returns the path of the chosen file as a string
    cartoonify(ImagePath)

def cartoonify(ImagePath):
    #Read the image
    uploadimage = cv2.imread(ImagePath)
    uploadimage = cv2.cvtColor(uploadimage, cv2.COLOR_BGR2RGB)
    print(uploadimage) #images in the form of numbers

    #check if the image is properly uploaded
    if uploadimage is None:
        print("No image found")
        sys.exit() #successful termination of the loop

    #Resizing the iimage to the desired size
    resize1 = cv2.resize(uploadimage, (960, 540))
    plt.imshow(resize1, cmap = 'gray')

    #converting the uploaded image to gray scale
    grayscaleimage = cv2.cvtColor(uploadimage, cv2.COLOR_BGR2GRAY)
    resize2 = cv2.resize(uploadimage, (960,540))
    plt.imshow(resize2, cmap='gray')

    #smoothening the grayscale image
    #To smoothen the image, we simply blur the image using medianblur option
    smoothgrayscaleimage = cv2.medianBlur(uploadimage, 5)
    resize3  = cv2.resize(smoothgrayscaleimage, (960,540))
    plt.imshow(resize3, cmap= 'gray')


    #Retriving the edges of the image
    getimageedge = cv2.adaptiveThreshold(smoothgrayscaleimage, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,9,9)
    resize4 = cv2.resize(getimageedge, (960,540))
    plt.imshow(resize4, cmap = 'gray')


    #MASKING THE IMAGE

    colorimage= cv2.bilateralFilter(uploadimage, 9,300, 300)
    resize5 = cv2.resize(colorimage, (960, 540))
    plt.imshow(resize5, cmap = 'gray')

    #Giving a cartoon effect
    cartoonimage = cv2.bitwise_and(colorimage, colorimage, mask = getimageedge)
    resize6 = cv2.resize(cartoonimage, (960, 540))
    plt.imshow(resize6, cmap= 'gray')


    #plotting all the transitions together  
    images = [resize1, resize2, resize3, resize4, resize5, resize6]
    fig, axes = plt.subplots(3,2, figsize=(8,8), subplot_kw={'xticks':[], 'yticks':[]}, gridspec_kw=dict(hspace=0.1, wspace=0.1))
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i], cmap='gray')
    plt.show()




