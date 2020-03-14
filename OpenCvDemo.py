#pip install opencv-python

import cv2
img = cv2.imread("img_42.jpg")
#cv2.imshow("result",img)
print(img)

#convert color
grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


#draw rectangle
#cv2.rectangle(image, start_point (x,y), end_point (width, height), color (bgr), thickness)
cv2.rectangle(img, (10,10), (100,100), (255,0,0), 5)

#cv2.imshow("grey",grey)
cv2.imshow("result", img)
