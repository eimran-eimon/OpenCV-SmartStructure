import numpy as np
import cv2
import math


def get_distance_between_red_lines(img, dilate_itr):
	org_img = img
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	edges = cv2.Canny(img, 50, 150, apertureSize=3)
	edges = cv2.dilate(edges, np.ones((1, 5), np.uint8), iterations=dilate_itr)
	min_line_length = img.shape[1] - 500
	lines = cv2.HoughLinesP(image=edges, rho=0.02, theta=np.pi / 180, threshold=10, lines=np.array([]),
							minLineLength=min_line_length, maxLineGap=10)
	if lines is None:
		return
	a, b, c = lines.shape
	lines_segment_y1 = []
	lines_segment_y2 = []
	for i in range(a):
		# print(f"{lines[i][0][1]}, {lines[i][0][3]}")
		# assuming red lines are almost horizontal
		# max difference between two end of line is less than 5 pixel
		if abs(lines[i][0][1] - lines[i][0][3]) < 5:
			lines_segment_y1.append(lines[i][0][1])
			lines_segment_y2.append(lines[i][0][3])
			# print(f"{lines[i][0][1]}, {lines[i][0][3]}")
			cv2.line(org_img, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), (0, 0, 255), 2, cv2.LINE_AA)

	y1_set = set(np.diff(sorted(lines_segment_y1)))
	y1_reduced_list = [y1 for y1 in y1_set if (y1 >= 15) and (y1 <= 30)]
	mean_y1 = np.mean(y1_reduced_list)

	y2_set = set(np.diff(sorted(lines_segment_y2)))
	y2_reduced_list = [y2 for y2 in y2_set if (y2 >= 15) and (y2 <= 30)]
	mean_y2 = np.mean(y2_reduced_list)
	mean_y = np.mean([mean_y1, mean_y2])
	print(f"Distance, 1ft={mean_y} px")
	if not math.isnan(mean_y) or mean_y is not None:
		cv2.imshow('detected_edges', org_img)

	return mean_y



