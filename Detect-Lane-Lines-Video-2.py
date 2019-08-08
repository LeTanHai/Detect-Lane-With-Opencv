import cv2
import numpy as np


cap = cv2.VideoCapture('test-video-2.mp4')

while True:
	ret,frame = cap.read()

	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

	yellow_up = np.array([48,255,255])
	yellow_low = np.array([18,94,140])

	white_up = np.array([255,255,255])
	white_low = np.array([180,180,180])

	mask = cv2.inRange(hsv,yellow_low,yellow_up)
	#mask = cv2.inRange(hsv,white_low,white_up)
	canny = cv2.Canny(mask, 75,150)

	# trả về các cặp tọa độ theo ma trận 2x2
	lines = cv2.HoughLinesP(canny,2,np.pi/180,100,np.array([]),minLineLength = 50,maxLineGap = 50)

	if lines is not None:
		for line in lines:
			x1,y1,x2,y2 = line.reshape(4)  # trả về 4 số riêng biệt
			cv2.line(frame,(x1,y1),(x2,y2),(0,255,255),5)

	cv2.imshow('result',frame)
	key = cv2.waitKey(5) & 0xFF
	if key == 27:
		break

cap.release()
cv2.destroyAllWindows()