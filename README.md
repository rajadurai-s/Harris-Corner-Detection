# Harris-Corner-Detection
A Python implementation of Harris Corner Detection algorithm with analysis of rotation invariancy, computation complexity and SNR calculation.

#Input
The code requires four arguments:
* img - Image file read with cv2 library
* k - Harris empirical constant which takes value between 0.04-0.06
* window_size - Size of the sliding window 
* threshold - Value of Harris response, above which a pixel is considered as a corner

#Output
The code returns 5 output files:
* corner_list - A .txt file containing the list of corner points in (x,y,r) format
* corner_img - Image with corners marked in blue dots
* t - Total computation time of the code
* t_array - An array of computation time for each pixel
* pix - An integer variable with total number of pixels

