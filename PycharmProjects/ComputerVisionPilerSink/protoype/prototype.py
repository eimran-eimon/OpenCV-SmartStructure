import sys
import cv2
import numpy as np
import xlsxwriter
from datetime import datetime
import os
import image_deskewd_and_get_distance
import math

# input video or live feed
cap = cv2.VideoCapture('pilevideo.mp4')

# mode for calc the 1ft px movement
auto_mode = None

# for saving the output result
# fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
# frame_width = int(cap.get(3))
# frame_height = int(cap.get(4))
# out = cv2.VideoWriter('output.mp4', fourcc, 20, (frame_width, frame_height), True)

# config data for the program
median_del_y = 0
frame_no = 0

# for selecting the ROI
rect = (0, 0, 0, 0)
line_1 = (0, 0)
line_2 = (0, 0)
startPoint = False
endPoint = False
drawing = False

# excel config
# Create a workbook and add a worksheet.
workbook = xlsxwriter.Workbook(f"data({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}).xlsx")
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
worksheet.write(row, col + 1, 'Piller Grounded', header_format)
row = row + 1

red_line_dist = None
prev_sinked = -math.inf
template = None
template_match_coord = []
no_of_template = 2
already_sinked = 0
sinked = 0
prev_top_left = -math.inf

# count no of click for manual distance calc
n = 0
# declare a list to append all the
# points on the image we clicked
prev_drawn_line_y = []
points = []
complete_input = None

# make this False, if the error is still persisted
draw_plots = True


def on_mouse(event, x, y, flags, params):
	if auto_mode is None:
		return

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
		if startPoint and not endPoint:
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


def release_resources():
	cv2.destroyAllWindows()
	cap.release()
	workbook.close()


while True:

	if auto_mode is None:
		frame = np.zeros((500, 500))
	else:
		ret, frame = cap.read()
		frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		if ret is False:
			continue

	org_frame = np.array(frame).copy()

	# if img_estimate(frame, 200) == 'light':
	# 	frame = adjust_brightness_contrast(frame, 0, 50)
	# elif img_estimate(frame, 100) == 'dark':
	# 	frame = adjust_brightness_contrast(frame, 50, 0)

	cv2.namedWindow('frame')
	cv2.setMouseCallback('frame', on_mouse)

	# drawing rectangle on mouse move
	if startPoint is True:
		cv2.rectangle(frame, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 0), 1)

	if startPoint is True and endPoint is True:
		if red_line_dist is None:
			selected_rectangle = org_frame[rect[1]:rect[3], rect[0]: rect[2]]
			# get distance between red lines
			# de-skewed and find distance
			if auto_mode is True:
				final_angle, rotated_image, max_scored_histogram = image_deskewd_and_get_distance.skew_correction(
					selected_rectangle)
				if draw_plots:
					heatmap = image_deskewd_and_get_distance.draw_plots(selected_rectangle, rotated_image, final_angle,
																		max_scored_histogram)
				else:
					heatmap = None

				red_line_dist = image_deskewd_and_get_distance.get_distance(max_scored_histogram, heatmap)

				if math.isnan(red_line_dist):
					release_resources()
					# out.release()
					os.execl(sys.executable, sys.executable, *sys.argv)

			elif auto_mode is False:
				red_line_dist = np.abs(rect[1] - rect[3])
				auto_mode = True
				startPoint = False
				endPoint = False
				continue

		template_h = np.abs(rect[1] - rect[3]) / no_of_template
		print(f"red dist = {red_line_dist}")
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

		w, h = template.shape[::-1]
		# print(f"DEL Y= {median_del_y}")
		# draw the rectangle box
		bottom_right = (top_left[0] + w, current_y + h)
		cv2.rectangle(frame[rect[1]:rect[3], rect[0]:rect[2]], (top_left[0], current_y), bottom_right, (255, 0, 0), 2)

		sinked = already_sinked + (median_del_y / red_line_dist)

		# draw an arrow
		# cv2.line(frame, (rect[0] - 50, rect[1]), (rect[0] - 50, int(rect[1] + median_del_y)), (0, 255, 255), 4)
		# cv2.arrowedLine(frame, (rect[0] - 50, int(rect[1] + median_del_y)), (rect[0] - 50, int(rect[3] + median_del_y)),
		# 				(0, 0, 255), 3)

		cv2.putText(frame, "Sinked: {:.2f}ft".format(sinked),
					(50, 30), cv2.FONT_HERSHEY_SIMPLEX,
					1, (0, 0, 255), 2)

		if sinked > prev_sinked:
			worksheet.write(row, col, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), center)
			worksheet.write(row, col + 1, sinked, center)
			prev_sinked = sinked
			# increase excel row number
			row = row + 1
		# count the frame no
		frame_no = frame_no + 1

	instruction_text = ""
	instruction_color = (0, 0, 0)
	if auto_mode is True and endPoint is False:
		instruction_text = "Please select the Region Of Interest"
		instruction_color = (0, 0, 255)
	elif auto_mode is False and red_line_dist is None:
		instruction_text = "Please select 1ft area on the Piller"
		instruction_color = (0, 255, 0)
	cv2.putText(frame, f"{instruction_text}",
				(50, 30), cv2.FONT_HERSHEY_SIMPLEX,
				0.8, instruction_color, 2)

	cv2.putText(frame, "*Press and Hold respective keys for 5s",
				(50, 80), cv2.FONT_HERSHEY_SIMPLEX,
				0.5, (250, 0, 20), 1)

	cv2.putText(frame, "Key r - Restart the program",
				(50, 130), cv2.FONT_HERSHEY_SIMPLEX,
				0.7, (250, 0, 20), 2)
	cv2.putText(frame, "Key ESC - Quit the program",
				(50, 180), cv2.FONT_HERSHEY_SIMPLEX,
				0.7, (250, 0, 20), 2)

	if auto_mode is None:
		cv2.putText(frame, "Key A - Automated calculation",
					(50, 230), cv2.FONT_HERSHEY_SIMPLEX,
					0.7, (250, 0, 20), 2)
		cv2.putText(frame, "Key M - Manual calculation",
					(50, 280), cv2.FONT_HERSHEY_SIMPLEX,
					0.7, (250, 0, 20), 2)

	cv2.imshow('frame', frame)
	# out.write(frame)
	if auto_mode is None:
		if cv2.waitKey(32) == ord('a'):
			auto_mode = True
		if cv2.waitKey(32) == ord('m'):
			auto_mode = False

	if cv2.waitKey(32) == ord('r'):
		release_resources()
		# out.release()
		os.execl(sys.executable, sys.executable, *sys.argv)
	# if 'ESC' is pressed, quit the program
	k = cv2.waitKey(32) & 0xff
	if k == 27:
		# release resources
		cv2.destroyAllWindows()
		cap.release()
		workbook.close()
		# out.release()
		break
