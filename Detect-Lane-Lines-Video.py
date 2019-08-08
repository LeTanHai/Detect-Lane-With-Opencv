import cv2
import numpy as np
import matplotlib.pyplot as plt

def find_lanes(image):
    height = canny.shape[0]
    #triagle = np.array([[(200,height),(1100,height),(550,250)]])
    triagle = np.array([[(0,300),(0,900),(1200,900),(1200,300)]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, triagle, (255,255,255))
    mask_image = cv2.bitwise_and(canny,canny,mask = mask)
    return mask_image

def display_lane(image,lane_line):
	if lane_line is not None:
		for line in lane_line:
			x1,y1,x2,y2 = line.reshape(4)
			cv2.line(image,(x1,y1),(x2,y2),(255,0,0),15)
	return image


image = cv2.VideoCapture('test-video.mp4')
#lane_image = np.copy(image)
while True:
    ret,lane_image = image.read()
    #canny = canny(lane_image)
    gray = cv2.cvtColor(lane_image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    canny = cv2.Canny(blur,50,150)
    lanes = find_lanes(canny)
# trả về tọa độ các cặp lane theo ma trận 2x2
    lane_line = cv2.HoughLinesP(lanes,2,np.pi/180,100,np.array([]),minLineLength = 50,maxLineGap = 4)
    result = display_lane(lane_image,lane_line)
    cv2.imshow('result',result)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

image.release()
cv2.destroyAllWindows()
#plt.imshow(image)
#plt.show()
