import cv2
import numpy as np

# input video or live feed
cap = cv2.VideoCapture('pilevideo.mp4')

# get the template input from user
template = cv2.imread('template.png', 0)
w, h = template.shape[::-1]

# for saving the output result
fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
out = cv2.VideoWriter('output.mp4', fourcc, 20, (frame_width, frame_height), True)

# config data for the program
total_del_y = 0
prev_top_left_y = 0
frame_no = 0
temp_prev_y_coord = np.zeros(10)
prev_current_y = -99999

# for selecting the ROI
rect = (0, 0, 0, 0)
startPoint = False
endPoint = False
drawing = False


def on_mouse(event, x, y, flags, params):
	global rect, startPoint, endPoint, drawing
	# get mouse click
	if event == cv2.EVENT_LBUTTONDOWN:
		if not startPoint:
			rect = (x, y, 0, 0)
			startPoint = True
			drawing = True

	elif event == cv2.EVENT_MOUSEMOVE:
		if drawing and startPoint is True:
			rect = (rect[0], rect[1], x, y)

	elif event == cv2.EVENT_LBUTTONUP:
		if not endPoint:
			rect = (rect[0], rect[1], x, y)
			endPoint = True
			drawing = False


while True:

	ret, frame = cap.read()
	frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	cv2.namedWindow('frame')
	cv2.setMouseCallback('frame', on_mouse)

	# drawing rectangle on mouse move
	if startPoint is True:
		cv2.rectangle(frame, (rect[0], rect[1]), (rect[2], rect[3]), (255, 0, 255), 2)

	if startPoint is True and endPoint is True:
		print(rect)
		# Perform match operations
		res = cv2.matchTemplate(frame_gray[rect[1]:rect[3], rect[0]:rect[2]], template, cv2.TM_SQDIFF_NORMED)

		# find the template's location in the video
		min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
		top_left = min_loc

		# remove noisy calculations
		# save data from previous ten frames for getting more robust Y coordinate
		if frame_no > 10:
			if abs(np.median(temp_prev_y_coord) - prev_current_y) < 5:
				temp_prev_y_coord[frame_no % 10] = top_left[1]
			else:
				temp_prev_y_coord[frame_no % 10] = prev_current_y
		else:
			temp_prev_y_coord[frame_no % 10] = top_left[1]

		# calculate the del Y (ten frames interval)
		if frame_no % 10 == 0 and frame_no != 0:
			# take the 50th percentile data
			# so that, some noisy coordinate can't affect the measurement
			del_y = top_left[1] - np.quantile(temp_prev_y_coord, 0.5)

			# assuming, del Y should be a small number between 0 to 1
			# and the only movement of the piller will be in the bottom direction
			if 0 <= del_y <= 1 and top_left[1] > prev_current_y:
				prev_current_y = top_left[1]
				total_del_y = total_del_y + del_y

		print(f"Current Y_coord= {top_left[1]}")
		# print(f"DEL Y= {total_del_y}")

		# draw the rectangle box
		bottom_right = (top_left[0] + w, top_left[1] + h)
		cv2.rectangle(frame[rect[1]:rect[3], rect[0]:rect[2]], top_left, bottom_right, 255, 2)

		# 15 pixels movement in Y direction = 1ft,
		# later we can generalized it by measuring the distance between red lines
		cv2.putText(frame, "Sinked: {:.2f}ft".format(total_del_y / 15),
					(50, 50), cv2.FONT_HERSHEY_SIMPLEX,
					1, (255, 0, 0), 3)
		# count the frame no
		frame_no = frame_no + 1

	cv2.imshow('frame', frame)
	out.write(frame)

	# if 'ESC' is pressed, quit the program
	k = cv2.waitKey(30) & 0xff
	if k == 27:
		# release resources
		cv2.destroyAllWindows()
		cap.release()
		break
