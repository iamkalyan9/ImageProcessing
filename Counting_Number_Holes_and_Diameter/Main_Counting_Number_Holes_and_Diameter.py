# Integrated code
# @author: Kalyan
# Date: 27-11-2021
# e-mail: kalyan9@yahoo.com


import time
import cv2 as cv
import numpy as np

start_time = time.time()
# load image as grayscale
cell1 = cv.imread("image_1.png",0)
# threshold image
ret,thresh_binary = cv.threshold(cell1,150, 255,cv.THRESH_BINARY)
# findcontours
contours, hierarchy = cv.findContours(image =thresh_binary , mode = cv.RETR_TREE,method = cv.CHAIN_APPROX_SIMPLE)

# create an empty mask
mask = np.zeros(cell1.shape[:2],dtype=np.uint8)

# loop through the contours
for i,cnt in enumerate(contours):
        # if the contour has no other contours inside of it
        if hierarchy[0][i][2] == -1 :
            
            # if the size of the contour is greater than a threshold
            if  cv.contourArea(cnt) <= 5050 and cv.contourArea(cnt) >= 1000:
                cv.drawContours(mask,[cnt], 0, (255), -1)                    
# display result
cv.imwrite('Testing_image.png', mask)

image = cv.imread('Testing_image.png')
# Gray, blur, adaptive threshold
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
blur = cv.GaussianBlur(gray, (3,3), 0)
thresh = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)[1]

# Morphological transformations
kernel = cv.getStructuringElement(cv.MORPH_RECT, (5,5))
opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel)

# Find contours
cnts = cv.findContours(opening, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
a = 0
z = 0
print("Units: cm")

for c in cnts:
    # Find perimeter of contour
    perimeter = cv.arcLength(c, True)
    #print('perimeter: ',perimeter)
    #perimeter = perimeter/10000
    # Perform contour approximation
    approx = cv.approxPolyDP(c, 0.04 * perimeter, True)

    # We assume that if the contour has more than a certain
    # number of verticies, we can make the assumption that the contour shape is a circle
    if len(approx) > 1: #6
        # Obtain bounding rectangle to get measurements
        x,y,w,h = cv.boundingRect(c)
        
        # Find measurements
        diameter = w/185

        # Find centroid
        M = cv.moments(c)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        # Draw the contour and center of the shape on the image
        #cv.rectangle(image,(x,y),(x+w,y+h),(0,255,0),4)
        cv.drawContours(image,[c], 0, (36,255,12), 4)
        cv.circle(image, (cX, cY), 15, (320, 159, 22), -1) 
        a=a+1
        z=z+35

        # Draw line and diameter information 
        cv.line(image, (x, y + int(h/2)), (x + w, y + int(h/2)), (156, 188, 24), 1)

        # After considering error precision
        cv.putText(image, "{:.1f} ".format(diameter), (cX - 1, cY - 6), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

cv.imshow("Final Image.png", image)
cv.imshow("Main Image.png", cell1)
print("--- %s seconds ---" % (time.time() - start_time))

cv.waitKey(0)
cv.destroyAllWindows()

