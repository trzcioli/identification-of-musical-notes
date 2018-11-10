import numpy as np
from matplotlib import pyplot as plt
import cv2 

fig = plt.figure
img = cv2.imread('Wlazlkotek_4_ubt.png') 
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) 
edges = cv2.Canny(gray,50,150,apertureSize = 3) 
lines = cv2.HoughLines(edges,1,np.pi/180, 200) 

lists = []
for line in lines:
	for r, theta in line: 
		a = np.cos(theta)
		b = np.sin(theta)
		x0 = a*r
		y0 = b*r
		x1 = int(x0 + 1000*(-b))
		y1 = int(y0 + 1000*(a))
		x2 = int(x0 - 1000*(-b))
		y2 = int(y0 - 1000*(a))
		lists.append([(x1, y1), (x2, y2)])
		cv2.line(img,(x1,y1), (x2,y2), (0,0,255),2)

print(lists)
cv2.imwrite('linesDetected.jpg', img) 
