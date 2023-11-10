#!/usr/bin/env python
# coding: utf-8

# In[1]:


import argparse
import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal as sig
import time


# # SOBEL MATRIX AND GAUSS MATRIX

# In[2]:


#Sobel matrices for derivative calculation and Gaussian matrix for denoising
#Instead of Sobel matrix, Prewitt and Robert matrices can also be used

# Sobel x-axis kernel
SOBEL_X = np.array((
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]), dtype="int32")

# Sobel y-axis kernel
SOBEL_Y = np.array((
    [-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1]), dtype="int32")

# Gaussian kernel
GAUSS = np.array((
    [1/16, 2/16, 1/16],
    [2/16, 4/16, 2/16],
    [1/16, 2/16, 1/16]), dtype="float64")


# # HARRIER CORNER DETECTION ALGORITHM

# In[3]:


def find_harris_corners(input_img, k, window_size, threshold):
    
    t_list=[]
    t1=time.perf_counter()
    corner_list = [] #to store the location of the corners in the image
    output_img = cv2.cvtColor(input_img.copy(), cv2.COLOR_BGR2GRAY) #converting rgb image to grayscale image
    output_img1 = input_img.copy() #creating a copy of the image to mark corners
    
    offset = int(window_size/2)
    y_range = input_img.shape[0] - offset
    x_range = input_img.shape[1] - offset
    
    
    dx = sig.convolve2d(output_img, SOBEL_X) # convolving with sobel filter on X-axis
    dy = sig.convolve2d(output_img, SOBEL_Y) # convolving with sobel filter on Y-axis
    # square of derivatives
    dx2 = np.square(dx)
    dy2 = np.square(dy)
    dxdy = dx*dy #cross filtering
    # gauss filter for all directions (x,y,cross axis)
    g_dx2 = sig.convolve2d(dx2, GAUSS)
    g_dy2 = sig.convolve2d(dy2, GAUSS)
    g_dxdy = sig.convolve2d(dxdy, GAUSS)
    
    pixel=[]
    i=1
    for y in range(offset, y_range):
        for x in range(offset, x_range):
            pixel.append(i)
            t3=time.perf_counter()
            #Values of sliding window
            start_y = y - offset
            end_y = y + offset + 1
            start_x = x - offset
            end_x = x + offset + 1
            
            #The variable names are representative to 
            #the variable of the Harris corner equation
            windowIxx = g_dx2[start_y : end_y, start_x : end_x]
            windowIxy = g_dxdy[start_y : end_y, start_x : end_x]
            windowIyy = g_dy2[start_y : end_y, start_x : end_x]
            
            #Sum of squares of intensities of partial derevatives 
            Sxx = windowIxx.sum()
            Sxy = windowIxy.sum()
            Syy = windowIyy.sum()

            #Calculate determinant and trace of the matrix
            det = (Sxx * Syy) - (Sxy**2)
            trace = Sxx + Syy
            
            #Calculate r for Harris Corner equation
            r = det - k*(trace**2) #single measure value
            
            #non-maximum suppression
            cv2.normalize(r, r, 0, 1, cv2.NORM_MINMAX)

            if r > threshold:
                corner_list.append([x, y, r])
                output_img1[y,x] = (0,0,255)
            t4=time.perf_counter()
            t_list.append(t4-t3)
            i+=1
    
    t2=time.perf_counter()
    return corner_list, output_img1,(t2-t1),t_list,pixel


# In[4]:


def show_img(img,bw=False):
    fig=plt.figure(figsize=(13,13))
    ax=fig.gca()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.imshow(img,cmap='Greys_r')
    plt.show()


# # RESULTS AND ANALYSIS

# In[38]:


img1=cv2.imread('ex1.png')
k=0.04
window_size=5
threshold=10000.00
corner_list, corner_img,t,t_array,pix= find_harris_corners(img1, k, window_size, threshold)
plt.figure(figsize=(13,13))
plt.subplot(121),plt.imshow(img1)
plt.title("Raw Image"), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(corner_img)
plt.title("Harris Corner Output"), plt.xticks([]), plt.yticks([])
print("Total time taken:",t,"seconds")


# # Computational complexity per pixel

# In[40]:


#computation time per pixel and window
plt.plot(pix,t_array)
plt.xlabel("Pixel number")
plt.ylabel("Computation time")


# # Calculating SNR

# In[24]:


def PSNR(original, calculated): 
    mse = np.mean((original - calculated) ** 2) 
    if(mse == 0):  # MSE is zero means no noise is present in the signal . 
                  # Therefore PSNR have no importance. 
        return 100
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse)) 
    return psnr

img1_gray=cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
original=cv2.cornerHarris(img1_gray,5,3,0.04)
corner_img_gray=cv2.cvtColor(corner_img, cv2.COLOR_BGR2GRAY)
snr=PSNR(original,corner_img_gray)
display(snr)


# # checking for invariance to rotation

# In[42]:


img=cv2.imread('ex1.png')
k=0.04
window_size=5
threshold=10000.00
img_rot=cv2.rotate(img,cv2.ROTATE_90_CLOCKWISE) #90 degree clockwise rotation
corner_list, corner_img,t,t_array,pix= find_harris_corners(img_rot, k, window_size, threshold)
plt.figure(figsize=(13,13))
plt.subplot(121),plt.imshow(img_rot)
plt.title("Raw Image"), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(corner_img)
plt.title("Harris Corner Output"), plt.xticks([]), plt.yticks([])
print("Total time taken:",t,"seconds")


# In[24]:


img=cv2.imread('ex1.png')
k=0.04
window_size=5
threshold=10000.00
img_rot=cv2.rotate(img,cv2.ROTATE_180) #180 degree rotation of the input image
corner_list, corner_img,t = find_harris_corners(img_rot, k, window_size, threshold)
plt.figure(figsize=(13,13))
plt.subplot(121),plt.imshow(img_rot)
plt.title("Raw Image"), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(corner_img)
plt.title("Harris Corner Output"), plt.xticks([]), plt.yticks([])
print("Total time taken:",t,"seconds")


# # Results for other test images

# In[25]:


img=cv2.imread('ex2.png')
k=0.04
window_size=5
threshold=10000.00
corner_list, corner_img,t = find_harris_corners(img, k, window_size, threshold)
plt.figure(figsize=(13,13))
plt.subplot(121),plt.imshow(img)
plt.title("Raw Image"), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(corner_img)
plt.title("Harris Corner Output"), plt.xticks([]), plt.yticks([])
print("Total time taken:",t,"seconds")


# In[26]:


img=cv2.imread('ex6.png')
k=0.04
window_size=5
threshold=10000.00
corner_list, corner_img,t = find_harris_corners(img, k, window_size, threshold)
plt.figure(figsize=(13,13))
plt.subplot(121),plt.imshow(img)
plt.title("Raw Image"), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(corner_img)
plt.title("Harris Corner Output"), plt.xticks([]), plt.yticks([])
print("Total time taken:",t,"seconds")


# In[27]:


img=cv2.imread('toy_image.jpg')
k=0.04
window_size=5
threshold=10000.00
corner_list, corner_img,t = find_harris_corners(img, k, window_size, threshold)
plt.figure(figsize=(13,13))
plt.subplot(121),plt.imshow(img)
plt.title("Raw Image"), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(corner_img)
plt.title("Harris Corner Output"), plt.xticks([]), plt.yticks([])
print("Total time taken:",t,"seconds")


# In[5]:


img4=cv2.imread('chess.png')
k=0.04
window_size=5
threshold=10000.00
corner_list, corner_img4,t,t_array,pix = find_harris_corners(img4, k, window_size, threshold)
plt.figure(figsize=(13,13))
plt.subplot(121),plt.imshow(img4)
plt.title("Raw Image"), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(corner_img4)
plt.title("Harris Corner Output"), plt.xticks([]), plt.yticks([])
print("Total time taken:",t,"seconds")


# In[29]:


img=cv2.imread('gfg.png')
k=0.04
window_size=5
threshold=10000.00
corner_list, corner_img,t = find_harris_corners(img, k, window_size, threshold)
plt.figure(figsize=(13,13))
plt.subplot(121),plt.imshow(img)
plt.title("Raw Image"), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(corner_img)
plt.title("Harris Corner Output"), plt.xticks([]), plt.yticks([])
print("Total time taken:",t,"seconds")

