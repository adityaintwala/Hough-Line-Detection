# -*- coding: utf-8 -*-
"""
Aditya Intwala

This is a script to demonstrate the implementation of Hough transform function to
detect lines from the image.

Input:
    img - Full path of the input image.
    num_rho - No. of Rhos in Rho-Theta Hough Space. Default 180.
    num_theta - No. of Thetas in Rho-Theta Hough Space. Default 180.
    bin_threshold - bin / vote values above or below the bin_threshold are shortlisted as lines. Default 150.

Note:
    Playing with the three input parameters is required to obtain desired lines in different images.
    
Returns:
    line_img - Image with the Lines drawn
    lines - List of lines in format (rho,theta,x1,y1,x2,y2)

"""

import argparse
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

def find_hough_lines(image, edge_image, num_rhos, num_thetas, bin_threshold):
  #image size
  img_height, img_width = edge_image.shape[:2]
  img_height_half = img_height / 2
  img_width_half = img_width / 2
  
  # Rho and Theta ranges
  diag_len = np.sqrt(np.square(img_height) + np.square(img_width))
  dtheta = 180 / num_thetas
  drho = (2 * diag_len) / num_rhos
  
  ## Thetas is bins created from 0 to 180 degree with increment of the provided dtheta
  thetas = np.arange(0, 180, step=dtheta)
  
  ## Rho ranges from -diag_len to diag_len where diag_len is the diagonal length of the input image
  rhos = np.arange(-diag_len, diag_len, step=drho)
  
  # Calculate Cos(theta) and Sin(theta) it will be required later on while calculating rho
  cos_thetas = np.cos(np.deg2rad(thetas))
  sin_thetas = np.sin(np.deg2rad(thetas))
  
  # Hough accumulator array of theta vs rho, (rho,theta)
  accumulator = np.zeros((len(rhos), len(thetas)))
  
  # Hough Space plot for the image.
  figure = plt.figure()
  hough_plot = figure.add_subplot()
  hough_plot.set_facecolor((0, 0, 0))
  hough_plot.title.set_text("Hough Space")
  
  # Iterate through pixels and if non-zero pixel process it for hough space
  for y in range(img_height):
    for x in range(img_width):
      if edge_image[y][x] != 0: #white pixel
        edge_pt = [y - img_height_half, x - img_width_half]
        hough_rhos, hough_thetas = [], [] 
        
        # Iterate through theta ranges to calculate the rho values
        for theta_idx in range(len(thetas)):
          # Calculate rho value
          rho = (edge_pt[1] * cos_thetas[theta_idx]) + (edge_pt[0] * sin_thetas[theta_idx])
          theta = thetas[theta_idx]
          
          # Get index of nearest rho value
          rho_idx = np.argmin(np.abs(rhos - rho))
          
          #increment the vote for (rho_idx,theta_idx) pair
          accumulator[rho_idx][theta_idx] += 1
          
          # Append values of rho and theta in hough_rhos and hough_thetas respectively for Hough Space plotting.
          hough_rhos.append(rho)
          hough_thetas.append(theta)
        
        # Plot Hough Space from the values
        hough_plot.plot(hough_thetas, hough_rhos, color="white", alpha=0.05)

  # accumulator, thetas, rhos are calculated for entire image, Now return only the ones which have higher votes. 
  # if required all can be returned here, the below code could be post processing done by the user.
  
  # Output image with detected lines drawn
  output_img = image.copy()
  # Output list of detected lines. A single line would be a tuple of (rho,theta,x1,y1,x2,y2) 
  out_lines = []
  
  for y in range(accumulator.shape[0]):
    for x in range(accumulator.shape[1]):
      # If number of votes is greater than bin_threshold provided shortlist it as a candidate line
      if accumulator[y][x] > bin_threshold:
        rho = rhos[y]
        theta = thetas[x]
        
        # a and b are intercepts in x and y direction
        a = np.cos(np.deg2rad(theta))
        b = np.sin(np.deg2rad(theta))
        
        x0 = (a * rho) + img_width_half
        y0 = (b * rho) + img_height_half
        
        # Get the extreme points to draw the line
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        
        # Plot the Maxima point on the Hough Space Plot
        hough_plot.plot([theta], [rho], marker='o', color="yellow")
        
        # Draw line on the output image
        output_img = cv2.line(output_img, (x1,y1), (x2,y2), (0,255,0), 1)
        
        # Add the data for the line to output list
        out_lines.append((rho,theta,x1,y1,x2,y2))

  # Show the Hough plot
  hough_plot.invert_yaxis()
  hough_plot.invert_xaxis()
  plt.show()
  
  return output_img, out_lines

def peak_votes(accumulator, thetas, rhos):
    """ Finds the max number of votes in the hough accumulator """
    idx = np.argmax(accumulator)
    rho = rhos[int(idx / accumulator.shape[1])]
    theta = thetas[idx % accumulator.shape[1]]

    return idx, theta, rho


def theta2gradient(theta):
    """ Finds slope m from theta """
    return np.cos(theta) / np.sin(theta)


def rho2intercept(theta, rho):
    """ Finds intercept b from rho """
    return rho / np.sin(theta)

def main():
    
    parser = argparse.ArgumentParser(description='Find Hough lines from the image.')
    parser.add_argument('image_path', type=str, help='Full path of the input image.')
    parser.add_argument('--num_rho', type=float, help='No. of Rhos')
    parser.add_argument('--num_theta', type=float, help='No. of Thetas')
    parser.add_argument('--bin_threshold', type=int, help='Pixel values above or below the bin_threshold are lines.')
    
    args = parser.parse_args()
    
    img_path = args.image_path
    num_rho = 180
    num_theta = 180
    bin_threshold = 150
    lines_are_white = True
    
    if args.num_rho:
        num_rho = args.num_rho
        
    if args.num_theta:
        num_theta = args.num_theta
    
    if args.bin_threshold:
        bin_threshold = args.bin_threshold
    
    input_img = cv2.imread(img_path)
    
    #Edge detection on the input image
    edge_image = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    ret, edge_image = cv2.threshold(edge_image, 120, 255, cv2.THRESH_BINARY_INV)
    #edge_image = cv2.Canny(edge_image, 100, 200)
    
    cv2.imshow('Edge Image', edge_image)
    cv2.waitKey(0)

    if edge_image is not None:
        
        print ("Detecting Hough Lines Started!")
        line_img, lines = find_hough_lines(input_img, edge_image, num_rho, num_theta, bin_threshold)
        
        cv2.imshow('Detected Lines', line_img)
        cv2.waitKey(0)
        
        line_file = open('lines_list.txt', 'w')
        line_file.write('rho, \t theta, \t x1 ,\t y1,  \t x2 ,\t y2 \n')
        for i in range(len(lines)):
            line_file.write(str(lines[i][0]) + ' , ' + str(lines[i][1]) + ' , ' + str(lines[i][2]) + ' , ' + str(lines[i][3]) + ' , ' + str(lines[i][4]) + ' , ' + str(lines[i][5]) + '\n')
        line_file.close()
                
        if line_img is not None:
            cv2.imwrite("lines_img.png", line_img)
    else:
        print ("Error in input image!")
            
    print ("Detecting Hough Lines Complete!")



if __name__ == "__main__":
    main()
