# ELECENG-712

Digital Imaging - Project 2

The purpose of this project is to develop a non-linear, feature-preserving spatial filter using the bilateral filtering discussed in the class (see attached PDF for equation).
Please also implement the Gaussian filtering and compare the results.

Notes:

  (1) For Gaussian filtering, you need to generate a global mask for the filter (make sure the total weights of your mask is one). For bilateral filtering, you need to         generate a mask for each pixel based on its local neighborhood. Again, the weights need to sum to one for each mask.
  
  (2) The neighborhood size for the mask may be adjusted, but starting from a 5*5 would be a good idea.
  
  (3) Two output images should be submitted: one for the Gaussian filtering and the other for the bilateral filtering, by using the sample image provided on Canvas.
  
  (4) Please also submit the source code with your output images.
