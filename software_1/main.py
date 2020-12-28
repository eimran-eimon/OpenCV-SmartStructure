import sys
import time

import cv2
import numpy as np
from datetime import datetime
import os
import math
import csv

# input video or live feed
cap = cv2.VideoCapture('pilevideo3.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)
print(fps)
# config data for the program
median_del_y = 0

# ROI selection data
rect = (0, 0, 0, 0)
startPoint = False
endPoint = False
drawing = False

# debug field names
debug_fields = ['Frame no', 'Del Y', 'Value', 'Median Del Y (20 frames)']
# csv field names
data_fields = ['Date-Time', 'Measurement (in ft)']
# name of the csv file
data_filename = f"data_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.csv"
debug_filename = f"debug_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.csv"

# config variables
template = None
red_line_dist = None

prev_sinked = -math.inf
prev_top_left = -math.inf

already_sinked = 0
sinked = 0

template_match_coord = []

# no of template
no_of_template = 3

max_pixel_movement = 5
max_displacement_per_frame = 2

start_time = time.time()
fps_show_interval = 1  # displays the frame rate every 1 second
counter = 0
fps = 0
frame_no = 1


def on_mouse(event, x, y, flags, params):
	global rect, startPoint, endPoint, drawing

	# get mouse clicks
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


# generate template's coordinates
def generate_templates(template_no):
	roi_h = abs(rect[1] - rect[3])
	offset_y = roi_h / template_no
	templates_coord = []

	for y in np.linspace(0, roi_h, template_no + 1).tolist()[:-1]:
		y1 = int(round(y)) + rect[1]
		y2 = int(round(y + offset_y)) + rect[1]
		templates_coord.append([(rect[0], y1), (rect[2], y2)])

	return templates_coord


# release all the resources
def release_resources():
	cv2.destroyAllWindows()
	cap.release()
	csv_file.close()


# instruction texts
def put_instruction_texts():
	cv2.putText(frame, f"{instruction_text}",
				(50, 30), cv2.FONT_HERSHEY_SIMPLEX,
				0.8, instruction_color, 2)
	cv2.putText(frame, "Key r - Restart the program",
				(50, 130), cv2.FONT_HERSHEY_SIMPLEX,
				0.7, (250, 0, 20), 2)
	cv2.putText(frame, "Key ESC - Quit the program",
				(50, 180), cv2.FONT_HERSHEY_SIMPLEX,
				0.7, (250, 0, 20), 2)


# writing to csv file
with open(data_filename, 'w') as csv_file, open(debug_filename, 'w') as debug_file:
	# creating a csv writer object
	csv_writer = csv.writer(csv_file)
	csv_writer.writerow(data_fields)

	# creating debug file
	debug_file_writer = csv.writer(debug_file)
	debug_file_writer.writerow(debug_fields)

	while True:
		ret, frame = cap.read()
		# If any of the frame is corrupted, skip all the code
		# and try to capture another frame
		if ret is False:
			continue

		frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		# mouse event's coordinates will be recorded from this window
		cv2.namedWindow('frame')
		cv2.setMouseCallback('frame', on_mouse)

		# draw rectangle on mouse move
		if startPoint is True:
			# try:
			# 	# try to zoom, if possible
			# 	image_to_show = np.copy(frame)
			# 	cropped = image_to_show[rect[1]:rect[3], rect[0]:rect[2]]
			# 	cv2.imshow('zoomed ROI', cv2.resize(cropped, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC))
			# except Exception as e:
			# 	print(str(e))

			cv2.rectangle(frame, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 0), 1)

		if startPoint is True and endPoint is True:
			if red_line_dist is None:
				red_line_dist = np.abs(rect[1] - rect[3])
				# make start point and end point False to draw another rectangle
				startPoint = False
				endPoint = False
				continue

			template_h = np.abs(rect[1] - rect[3]) / no_of_template
			print(f"Calculated distance = {red_line_dist}")

			if template is None or (median_del_y + 10 > math.floor(template_h / 2)):
				# reset top left coordinates
				prev_top_left = 0
				# reset median_del_y for the current template
				median_del_y = 0

				# clear all the previous template's coordinates
				template_match_coord.clear()

				# save data of the previous template
				already_sinked = sinked

				# generate template
				template_coord = generate_templates(no_of_template)

				# select the upper template
				upper_template = template_coord[0]
				# extract the template for visualizing purpose
				template = frame_gray[upper_template[0][1]:upper_template[1][1], upper_template[0][0]:upper_template[1][0]]
				# show the template
				cv2.imshow('template', template)

			# Perform the match operations
			res = cv2.matchTemplate(frame_gray[rect[1]:rect[3], rect[0]:rect[2]], template, cv2.TM_SQDIFF_NORMED)

			# find the template's location in the video
			min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
			top_left = min_loc

			current_y = top_left[1]
			# print("min val", min_val)
			# print(f"Y-> {current_y}, prev_y-> {prev_top_left}")
			# find the next best match
			if abs(current_y - prev_top_left) > max_displacement_per_frame:
				match_result = np.array(res).flatten()
				sorted_match_result_idx = np.argsort(match_result)
				for idx in sorted_match_result_idx:
					if idx >= prev_top_left:
						# change the coord to next best match
						current_y = idx
						min_val = match_result[idx]
						# print(f"changed Y-> {current_y}")
						break

			if abs(current_y - prev_top_left) < max_pixel_movement:
				template_match_coord.append(current_y)
				# save the previous left Y coordinate to calculate noise in the data
				prev_top_left = current_y
				debug_file_writer.writerows([[frame_no, current_y, min_val, median_del_y]])

			# track the median of the last 20 coordinate
			# this "median_del_y" is the grounded measurement for the current template
			# "median_del_y" also used in resetting the template
			# data updated interval = 20 frames
			if len(template_match_coord) == 20:
				median_del_y = np.median(template_match_coord)
				template_match_coord.clear()
			sinked = already_sinked + (median_del_y / red_line_dist)

			# draw template match in the ROI
			w, h = template.shape[::-1]
			bottom_right = (top_left[0] + w, current_y + h)
			cv2.rectangle(frame[rect[1]:rect[3], rect[0]:rect[2]], (top_left[0], current_y), bottom_right, (255, 0, 0), 2)

			cv2.putText(frame, "Sinked: {:.2f}ft".format(sinked), (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

			# watch for change in measurement
			if sinked > prev_sinked:
				# write to the csv file
				csv_writer.writerows([[datetime.now().strftime('%Y-%m-%d %H:%M:%S'), round(sinked, 4)]])
				prev_sinked = sinked

		# change the instruction text and color according to the UX logic
		instruction_text = ""
		instruction_color = (0, 0, 0)

		if red_line_dist is None:
			instruction_text = "Please select 1ft area on the Piller"
			instruction_color = (0, 255, 0)
		elif template is None and red_line_dist > 0:
			instruction_text = "Please select the Region Of Interest"
			instruction_color = (0, 0, 255)

		# put instruction texts on the screen
		put_instruction_texts()

		# calc FPS
		counter += 1
		if (time.time() - start_time) > fps_show_interval:
			fps = counter / (time.time() - start_time)
			counter = 0
			start_time = time.time()

		cv2.putText(frame, f"Frame No: {frame_no}, FPS: {round(fps)}",
					(50, 80), cv2.FONT_HERSHEY_SIMPLEX,
					0.8, (0, 0, 0), 2)

		# show the resultant frame
		cv2.imshow('frame', frame)
		frame_no += 1
		# get the key input value
		k = cv2.waitKey(32) & 0xff

		if k == ord('r'):
			release_resources()
			os.execl(sys.executable, sys.executable, *sys.argv)

		# if 'ESC' is pressed, quit the program
		if k == 27:
			# release resources
			release_resources()
			break
