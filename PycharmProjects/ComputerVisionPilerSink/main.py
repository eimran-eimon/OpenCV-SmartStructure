import numpy as np
import cv2

cap = cv2.VideoCapture('pilevideo.mp4')

# params for ShiTomasi corner detection
feature_params = dict(maxCorners=100, qualityLevel=0.4, minDistance=7, blockSize=7)

# Parameters for lucas kanade optical flow
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Predefined Metadata
camera_calibration = 7  # depend on the camera's distance from the targeted object
tracker_reset = 150  # Reset after 150 frames

dist_per_pixel = 0.305 / (25 * camera_calibration)
total_sink = 0
frame_no = 0

# Create some random colors
color = np.random.randint(0, 255, (100, 3))

# Take first frame and find corners in it
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)

# for saving the result
fourcc = cv2.VideoWriter_fourcc('X','V','I','D')
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

out = cv2.VideoWriter('output.mp4', fourcc, 20, (frame_width, frame_height), True)

while True:
	ret, frame = cap.read()
	frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# reset the tracker after 150 frame
	if frame_no % tracker_reset == 0:
		p0 = cv2.goodFeaturesToTrack(frame_gray, mask=None, **feature_params)
	# calculate optical flow
	p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

	good_new = p1[st == 1]
	good_old = p0[st == 1]

	sum_of_all_dy = 0
	no_of_point = 1
	# draw the tracks
	for i, (new, old) in enumerate(zip(good_new, good_old)):
		a, b = new.ravel()
		c, d = old.ravel()

		# print(f"old-x:{a}, new-x: {c}\n")
		# print(f"old-y:{d}, new-y: {b}\n")

		# only count, if the pixel has only vertical motion
		if (b - d) > 0 and abs(a - c) < 0.1:
			sum_of_all_dy = sum_of_all_dy + abs(b - d)
			no_of_point = no_of_point + 1
			mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
			frame = cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)

	avg_dy = sum_of_all_dy / no_of_point
	print(f"average pixels dy: {avg_dy}")
	total_sink = total_sink + avg_dy
	img = cv2.add(frame, mask)

	# result on the screen
	cv2.putText(img, "Sinked: {:.2f}ft".format(total_sink * dist_per_pixel * 3.28),
				(50, 50), cv2.FONT_HERSHEY_SIMPLEX,
				1, (255, 0, 0), 3)
	cv2.resize(img, (600, 600))
	cv2.imshow('frame', img)
	k = cv2.waitKey(30) & 0xff
	if k == 27:
		break

	# Now update the previous frame and previous points
	old_gray = frame_gray.copy()
	p0 = good_new.reshape(-1, 1, 2)
	frame_no = frame_no + 1

cv2.destroyAllWindows()
cap.release()
