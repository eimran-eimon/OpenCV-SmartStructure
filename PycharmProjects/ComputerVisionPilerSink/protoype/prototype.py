import sys
import cv2
import numpy as np
import xlsxwriter
from datetime import datetime
import os
import get_red_line_distance
import math

# input video or live feed
cap = cv2.VideoCapture('pilevideo.mp4')

# for saving the output result
fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
out = cv2.VideoWriter('output.mp4', fourcc, 20, (frame_width, frame_height), True)

# config data for the program
total_del_y = 0
frame_no = 0

# for selecting the ROI
rect = (0, 0, 0, 0)
startPoint = False
endPoint = False
drawing = False

# excel config
# Create a workbook and add a worksheet.
workbook = xlsxwriter.Workbook('data.xlsx')
worksheet = workbook.add_worksheet()
worksheet.set_column(0, 4, 30)
header_format = workbook.add_format({
	'bold': True,
	'text_wrap': True,
	'align': 'center',
	'fg_color': '#D7E4BC',
	'border': 1})
# Start from the first cell. Rows and columns are zero indexed.
row = 0
col = 0
# Add a bold format to use to highlight cells.
center = workbook.add_format({'align': 'center'})
worksheet.write(row, col, 'Date-Time', header_format)
worksheet.write(row, col + 1, 'Piller Sinked', header_format)
row = row + 1

red_line_dist = None


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


def generate_templates(no_of_template):
	roi_h = abs(rect[1] - rect[3])
	offset_y = roi_h / no_of_template
	templates_coord = []

	for y in np.linspace(0, roi_h, no_of_template + 1).tolist()[:-1]:
		y1 = int(round(y)) + rect[1]
		y2 = int(round(y + offset_y)) + rect[1]
		templates_coord.append([(rect[0], y1), (rect[2], y2)])

	return templates_coord


def adjust_brightness_contrast(img):
	return img


while True:

	ret, frame = cap.read()
	frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	cv2.namedWindow('frame')
	cv2.setMouseCallback('frame', on_mouse)

	# drawing rectangle on mouse move
	if startPoint is True:
		cv2.rectangle(frame, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 0), 2)

	if startPoint is True and endPoint is True:

		if red_line_dist is None:
			# get distance between red lines
			full_rectangle = frame[0:rect[3], rect[0]:rect[2]]
			red_line_dist = get_red_line_distance.get_distance_between_red_lines(full_rectangle, 1)

			# try one more time with adjusted image
			if math.isnan(red_line_dist) or red_line_dist is None:
				# adjust brightness and contrast
				adjust_rect = adjust_brightness_contrast(full_rectangle)
				red_line_dist = get_red_line_distance.get_distance_between_red_lines(adjust_rect, 3)
				if math.isnan(red_line_dist) or red_line_dist is None:
					# restart the program
					os.execl(sys.executable, sys.executable, *sys.argv)

		if frame_no % 1000 == 0:
			# reset constrains
			temp_prev_y_coord = np.zeros(10)
			prev_current_y = -999999
			prev_top_left_y = 0

			template_coord = generate_templates(2)
			"""for temp_rect in template_coord:
				cv2.rectangle(frame, (temp_rect[0][0], temp_rect[0][1]), (temp_rect[1][0], temp_rect[1][1]), (0, 0, 255), 2)"""

			# select the upper one
			upper_template = template_coord[0]
			# extract the template
			template = frame_gray[upper_template[0][1]:upper_template[1][1], upper_template[0][0]:upper_template[1][0]]

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

		# print(f"Current Y_coord= {top_left[1]}")
		# print(f"DEL Y= {total_del_y}")

		# draw the rectangle box
		w, h = template.shape[::-1]
		bottom_right = (top_left[0] + w, top_left[1] + h)
		cv2.rectangle(frame[rect[1]:rect[3], rect[0]:rect[2]], top_left, bottom_right, (255, 0, 0), 2)

		sinked = total_del_y / red_line_dist
		cv2.putText(frame, "Sinked: {:.2f}ft".format(sinked),
					(50, 50), cv2.FONT_HERSHEY_SIMPLEX,
					1, (255, 0, 0), 2)

		worksheet.write(row, col, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), center)
		worksheet.write(row, col + 1, sinked, center)

		# count the frame no
		frame_no = frame_no + 1
		# increase excel row number
		row = row + 1

	if startPoint is False and endPoint is False:
		cv2.putText(frame, "Please draw a rectangle",
					(50, 60), cv2.FONT_HERSHEY_SIMPLEX,
					1, (255, 0, 0), 2)
	cv2.putText(frame, "Press r - Restart the program",
				(50, 100), cv2.FONT_HERSHEY_SIMPLEX,
				0.8, (255, 200, 0), 2)
	cv2.putText(frame, "Press ESC - Quit the program",
				(50, 140), cv2.FONT_HERSHEY_SIMPLEX,
				0.8, (255, 200, 0), 2)

	cv2.imshow('frame', frame)
	out.write(frame)

	if cv2.waitKey(30) == ord('r'):
		os.execl(sys.executable, sys.executable, *sys.argv)
	# if 'ESC' is pressed, quit the program
	k = cv2.waitKey(30) & 0xff
	if k == 27:
		# release resources
		cv2.destroyAllWindows()
		cap.release()
		workbook.close()
		break
