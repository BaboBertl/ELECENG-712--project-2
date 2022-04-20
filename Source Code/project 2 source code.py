import cv2
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg

#open the image, reads image as is from source, as a float
img = cv2.imread("C:/Users/Berto/p2input.jpg", cv2.IMREAD_UNCHANGED).astype(np.float32)/255.0


### broke the code up into two different sections as i worked on them 
### individually and then combined them together at the end


#create a function to apply a kernel to an image
def convolution(image, kernel):
    """
    'image' is the input image,
    'kernel' has to be defined else where and then inserted here
    'average' optional parameter, will be false unless stated otherwise

    """
    
    #define the boundaries of image
    row,col = image.shape
    
    #define the boundaries of kernel
    m,n = kernel.shape
    new = np.zeros((row+m-1, col+n-1))
    
    #doing integer division
    n = n//2
    m = m//2
    
    #making an array of zeros with the original shape of the input image
    filtered_image = np.zeros(image.shape)
    new[m:new.shape[0]-m,n:new.shape[1]-n] = image
    
    #create nested loop appying the kernel to image
    for i in range (m, new.shape[0]-m):
        for j in range (n, new.shape[1]-n):
            temp = new[i-m:i+m+1, j-m:j+m+1]
            result = temp*kernel
            filtered_image[i-m, j-n] = result.sum()       
    
    return filtered_image


#create a function to define kernel size
def gaussian_filter (m, n, sigma_radius):
    """
    'm' size of mask in y-direction,
    'n' size of mask in x-direction
    'sigma_radius' is the desired radius

    """
    
    #creates new array of given shape, filled with zeros
    gaussian_filter = np.zeros((m,n))
    
    #doing integer division
    m=m//2
    n=n//2
    
    #create nested loop to generate equation that defines gaussian filter
    for x in range (-m, m+1):
        for y in range (-n, n+1):
            x1 = 2*np.pi*(sigma_radius**2)
            x2 = np.exp(-(x**2+y**2)/(2*sigma_radius**2))
            gaussian_filter[x+m, y+n] = (1/x1)*x2
            
    return gaussian_filter


### end of code defining gaussian filter, start of code defining bilateral 
### filtering

#create a function to generate equation that defines bilateral filter
def bilateral_filter(image, sigma_s, sigma_r, reg_constant=1e-9):
    """
    'image' is the input image,
    'sigma_s' location parameter,
    'sigma_r' intensity parameter,
    'reg_constant' is the regularization constant

    """
    
    #make a Gaussian function taking the squared radius
    gaussian = lambda r2, sigma:(np.exp( -0.5*r2/sigma**2)*3).astype(int)*1.0/3.0

    #define the window width to be the 5 time the spatial standard deviation to 
    #be sure that most of the spatial kernel is actually captured
    window_width = int( 5*sigma_s+1 )

    #initialize the results and sum of weights to very small values for
    #numerical stability
    weighted_sum = np.ones(image.shape )*reg_constant
    result  = image*reg_constant

    #combine the result by shifting the image across the window in both the 
    #horizontal and vertical directions. in the inner loop, calculate the 
    #two weights and combine the weight sum and the unnormalized result image
    for shift_x in range(-window_width, window_width+1):
        for shift_y in range(-window_width, window_width+1):
            
            #calculate the spatial weight
            spatial_weight = gaussian(shift_x**2+shift_y**2, sigma_s)

            #shift by the offsets
            offset = np.roll(image, [shift_y, shift_x], axis = [0,1])

            #calculate the total weight
            total_weight = spatial_weight*gaussian((offset-image)**2, sigma_r)

            #find the results using assignment operator
            result += offset*total_weight
            weighted_sum += total_weight

    #normalize the result and end function
    return result/weighted_sum

#applying function to image to get gaussian filtered image
gaussian_kernel = gaussian_filter(5, 5, 2)
gaussian_filtered_img = convolution(img, gaussian_kernel)

#applying function to image to get bilateral filtered image
bilateral_filtered_img = bilateral_filter(img, 10, 0.3)

#to display images
plt.imshow(img, cmap='gray')
plt.figure()
plt.imshow(gaussian_filtered_img, cmap='gray')
plt.figure()
plt.imshow(bilateral_filtered_img, cmap='gray')

#to save images
mpimg.imsave("original image.jpg", img)
mpimg.imsave("gaussian filtered, sigma = 2.jpg", gaussian_filtered_img)
mpimg.imsave("bilateral filtered, sigma s = 10, sigma r = 0.3.jpg", bilateral_filtered_img)


