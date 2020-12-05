import sys
import cv2
import numpy as np
import xlsxwriter
from datetime import datetime
import os
import get_red_line_distance
import math

# input video or live feed
cap = cv2.VideoCapture('pilevideo2.mp4')

# for saving the output result
fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
out = cv2.VideoWriter('output.mp4', fourcc, 20, (frame_width, frame_height), True)

# config data for the program
median_del_y = 0
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
prev_sinked = -999999
template = None
template_match_coord = []
no_of_template = 2
already_sinked = 0
sinked = 0
prev_top_left = -math.inf


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


def generate_templates(template_no):
	roi_h = abs(rect[1] - rect[3])
	offset_y = roi_h / template_no
	templates_coord = []

	for y in np.linspace(0, roi_h, template_no + 1).tolist()[:-1]:
		y1 = int(round(y)) + rect[1]
		y2 = int(round(y + offset_y)) + rect[1]
		templates_coord.append([(rect[0], y1), (rect[2], y2)])

	return templates_coord


def adjust_brightness_contrast(img, brightness, contrast):
	img = np.int16(img)
	img = img * (contrast / 127 + 1) - contrast + brightness
	img = np.clip(img, 0, 255)
	img = np.uint8(img)
	return img


def img_estimate(img, threshold):
	is_light = np.mean(img) > threshold
	return 'light' if is_light else 'dark'


while True:

	ret, frame = cap.read()
	org_frame = np.array(frame).copy()
	frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	if img_estimate(frame, 200) == 'light':
		frame = adjust_brightness_contrast(frame, 0, 50)
	elif img_estimate(frame, 100) == 'dark':
		frame = adjust_brightness_contrast(frame, 50, 0)

	cv2.namedWindow('frame')
	cv2.setMouseCallback('frame', on_mouse)

	# drawing rectangle on mouse move
	if startPoint is True:
		cv2.rectangle(frame, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 0), 2)

	if startPoint is True and endPoint is True:

		if red_line_dist is None:
			# get distance between red lines
			full_rectangle = org_frame[0:frame_height, rect[0] - 10: rect[2] + 10]
			# adjust brightness and contrast
			adjust_rect = adjust_brightness_contrast(full_rectangle, 100, 50)
			red_line_dist = get_red_line_distance.get_distance_between_red_lines(adjust_rect, 2)

			# try one more time with adjusted image
			if math.isnan(red_line_dist) or red_line_dist is None:
				red_line_dist = get_red_line_distance.get_distance_between_red_lines(full_rectangle, 1)
				if math.isnan(red_line_dist) or red_line_dist is None:
					# restart the program
					os.execl(sys.executable, sys.executable, *sys.argv)

		template_h = abs(rect[1] - rect[3]) / no_of_template
		# print(f"Template h = {template_h}")
		if template is None or (median_del_y + 5 > math.floor(template_h / 2)):
			already_sinked = sinked
			prev_top_left = 0
			median_del_y = 0
			noisy_coord_count = 0
			template_match_coord.clear()
			template_coord = generate_templates(no_of_template)
			# select the upper one
			upper_template = template_coord[0]
			# extract the template
			template = frame_gray[upper_template[0][1]:upper_template[1][1], upper_template[0][0]:upper_template[1][0]]
			cv2.imshow('template', template)

		# Perform match operations
		res = cv2.matchTemplate(frame_gray[rect[1]:rect[3], rect[0]:rect[2]], template, cv2.TM_SQDIFF_NORMED)

		# find the template's location in the video
		min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
		top_left = min_loc

		# change the coord to next best match
		current_y = top_left[1]

		if (prev_top_left > current_y) or (abs(current_y - prev_top_left) > 5):
			match_result = np.array(res).flatten()
			sorted_match_result_idx = np.argsort(match_result)
			for idx in sorted_match_result_idx:
				if idx >= prev_top_left:
					current_y = idx
					break

		if abs(current_y - prev_top_left) < 5:
			template_match_coord.append(current_y)
			prev_top_left = current_y

		if len(template_match_coord) == 20:
			median_del_y = np.median(template_match_coord)
			template_match_coord.clear()

		# print(f"Current Y_coord= {current_y}")
		# print(f"Prev Y_coord= {prev_current_y}")
		w, h = template.shape[::-1]
		print(f"DEL Y= {median_del_y}")
		# draw the rectangle box
		bottom_right = (top_left[0] + w, current_y + h)
		# print(f"bottom right = {bottom_right[1] + 5}, and end = {rect[3]}")
		cv2.rectangle(frame[rect[1]:rect[3], rect[0]:rect[2]], (top_left[0], current_y), bottom_right, (255, 0, 0), 2)

		sinked = already_sinked + (median_del_y / red_line_dist)

		# draw an arrow
		cv2.line(frame, (rect[0] - 50, rect[1]), (rect[0] - 50, int(rect[1] + median_del_y)), (0, 255, 255), 4)
		cv2.arrowedLine(frame, (rect[0] - 50, int(rect[1] + median_del_y)), (rect[0] - 50, int(rect[3] + median_del_y)),
						(0, 0, 255), 3)

		cv2.putText(frame, "Sinked: {:.2f}ft".format(sinked),
					(50, 50), cv2.FONT_HERSHEY_SIMPLEX,
					1, (255, 0, 0), 2)

		# print(f"alreday sinked = {already_sinked}")
		if sinked > prev_sinked:
			worksheet.write(row, col, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), center)
			worksheet.write(row, col + 1, sinked, center)
			prev_sinked = sinked
			# increase excel row number
			row = row + 1
		# count the frame no
		frame_no = frame_no + 1

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
		out.release()
		break
