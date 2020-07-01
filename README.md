# Hough-Line-Detection
Implementation of Simple Hough Line Detection Algorithm in Python.\
This is based on paper [Use of the Hough Transformation To Detect Lines and Curves in Pictures](/Paper/HoughTransformPaper.pdf) by Richard O. Duda and Peter E. Hart.\

## Hough Space
A line can be represented in Cartesian Space by the following equation,
&nbsp; &nbsp; &nbsp; &nbsp; y = m * x + b &nbsp; &nbsp; &nbsp; &nbsp; where, m = gradient / slope of line.
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; b = y-intercept.
So given some set of points in binary image we can find lines connecting these points in Image Space. Lines in Cartesian Image Space will intersect at a point in m - b Parameter Space as shown in the figure.

![m - b Parameter Space](/images/m-b_space.png)

But this fails for vertical lines, i.e. m = 0. So we use Hough Space instead of m - b Parameter Space.
A line can be represented in polar form as show in the figure,

![rho - theta Parameter Space](/images/rho-theta_space.png)

The line from origin with distance rho has a slope of sin(theta) / cos(theta). The line of interest which is perpendicular to it will have negative reciprocal slope i.e. -cos(theta) / sin(theta).
The y-intercept of line is sin(theta) = rho / b. Thus inserting m = -cos(theta) / sin(theta) and b = rho / sin(theta) in the equation of line we get, 
&nbsp; &nbsp; &nbsp; &nbsp; rho = x * cos(theta) + y * sin(theta) &nbsp; &nbsp; &nbsp; &nbsp; where, rho = distance from origin to line.
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; theta = angle from origin to line.
														   
A Hough space is rho - theta space, Lines in Cartesian Image Space will intersect at a point in rho - theta Parameter Space.

![Hough Space](/images/HoughSpace_ex3.png)

## Usage
''' python find_hough_lines.py ./images/ex1.png --num_rho 180 --num_theta 180 --bin_threshold 150 '''

### Input
The script requires one positional argument and 3 optional parameters:
* image_path - Complete path to the image file for line detection.
* num_rho - No. of Rhos in Rho-Theta Hough Space. Default 180.
* num_theta - No. of Thetas in Rho-Theta Hough Space. Default 180.
* bin_threshold - bin / vote values above or below the bin_threshold are shortlisted as lines. Default 150.

### Output
The output of the script would be two files:
* lines.txt - File containing list of lines in format (rho,theta,x1,y1,x2,y2)
* line_img.png - Image with the Lines drawn in Green color.

## Samples
Sample Input Image  |  Sample Hough Space  |  Sample Output Image
:------------------:|:--------------------:|:--------------------:
![Sample Input Image](/images/ex1.png)  |  ![Sample Hough Space](/images/HoughSpace_ex1.png)  |  ![Sample Output Image](/images/output_ex1.png)
![Sample Input Image](/images/ex2.png)  |  ![Sample Hough Space](/images/HoughSpace_ex2.png)  |  ![Sample Output Image](/images/output_ex2.png)
![Sample Input Image](/images/ex3.png)  |  ![Sample Hough Space](/images/HoughSpace_ex3.png)  |  ![Sample Output Image](/images/output_ex3.png)

## Limitation
Playing with the three input parameters is required to obtain desired lines in different images so its not adaptive. Also the code can be little more optimized.


