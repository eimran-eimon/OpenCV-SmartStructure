# import the necessary packages
import dlib
from imutils.video import FileVideoStream
from imutils.video import VideoStream
import numpy as np
import time
import cv2

markerZero = False  # is marker zero detected
markerOne = False  # is marker one detected
arucoMarkerSizeInFt = 0.575  # 6.9 inch

markerZeroCoordinates = []  # save the path of the marker zero
markerOneCoordinates = []  # save the path of the marker one
tracker = []
# displacement in pixel
total_displacement_in_px = 0
# calculate and save marker size as long as program run to get more accurate data
marker_size = []
marker_labels = []
# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = FileVideoStream("test_video1.mp4").start()
# vs = VideoStream(0).start()
time.sleep(2.0)


# instruction texts
def default_instruction_texts():
	cv2.putText(frame, "Key r - Restart the program", (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (250, 0, 20), 2)
	cv2.putText(frame, "Key ESC - Quit the program", (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (250, 0, 20), 2)


def measure_displacement():
	if len(markerZeroCoordinates) < 2 or len(markerOneCoordinates) < 2:
		return 0
	marker_zero_total_displacement = markerZeroCoordinates[-1] - markerZeroCoordinates[0]
	marker_one_total_displacement = markerOneCoordinates[-1] - markerOneCoordinates[0]
	total_displacement = np.mean([marker_zero_total_displacement, marker_one_total_displacement])
	return total_displacement


# Assuming two markers have the same motion in Y-axis
def copy_displacement_of_the_other_marker(detected_marker_id):
	if len(markerZeroCoordinates) < 2 or len(markerOneCoordinates) < 2:
		return
	if detected_marker_id == 0:
		last_displacement = markerZeroCoordinates[-1] - markerZeroCoordinates[-2]
		markerOneCoordinates.append(markerOneCoordinates[-1] + last_displacement)
	elif detected_marker_id == 1:
		last_displacement = markerOneCoordinates[-1] - markerOneCoordinates[-2]
		markerZeroCoordinates.append(markerZeroCoordinates[-1] + last_displacement)


# Assuming the Pile will only go down in Y-axis
def save_displacement(detected_marker_id, current_y):
	if detected_marker_id == 0:
		if len(markerZeroCoordinates) == 0 or (current_y > markerZeroCoordinates[-1]):
			markerZeroCoordinates.append(current_y)
			return True
	elif detected_marker_id == 1:
		if len(markerOneCoordinates) == 0 or (current_y > markerOneCoordinates[-1]):
			markerOneCoordinates.append(current_y)
			return True
	else:
		return False


while True:
	# grab the frame from the threaded video stream
	# and resize it to have a maximum width of 1000 pixels
	frame = vs.read()
	# frame = imutils.resize(frame, width=1000)
	
	if frame is None:
		break
	
	key = cv2.waitKey(1) & 0xFF
	
	if key == ord("0") and markerZero is False:
		markerZero = True
		box = cv2.selectROI("Frame", frame, fromCenter=False, showCrosshair=True)
		print(box)
		t = dlib.correlation_tracker()
		rect = dlib.rectangle(int(box[0]), int(box[1]), int(box[0]) + int(box[2]), int(box[1]) + int(box[3]))
		# markerZeroCoordinates.append((int(box[1]) + (int(box[1]) + int(box[3]))) / 2)
		marker_size.append(int(box[3]))
		t.start_track(frame, rect)
		tracker.append(t)
		marker_labels.append(0)
	
	if key == ord("1") and markerOne is False:
		markerOne = True
		box = cv2.selectROI("Frame", frame, fromCenter=False, showCrosshair=True)
		t = dlib.correlation_tracker()
		rect = dlib.rectangle(int(box[0]), int(box[1]), int(box[0]) + int(box[2]), int(box[1]) + int(box[3]))
		# markerOneCoordinates.append((int(box[1]) + (int(box[1]) + int(box[3]))) / 2)
		marker_size.append(int(box[3]))
		t.start_track(frame, rect)
		tracker.append(t)
		marker_labels.append(1)
	
	if markerZero and markerOne:
		# loop over each of the trackers
		for (t, l) in zip(tracker, marker_labels):
			# update the tracker and grab the position of the tracked
			# object
			t.update(frame)
			pos = t.get_position()
			# unpack the position object
			startX = int(pos.left())
			startY = int(pos.top())
			endX = int(pos.right())
			endY = int(pos.bottom())
			is_saved = save_displacement(l, (startY + endY) / 2)
			total_displacement_in_px = measure_displacement()
			if True:
				# draw the bounding box from the correlation object tracker
				cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
				cv2.putText(frame, str(l), (startX, startY - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
	
	if markerZero is False:
		cv2.putText(frame, f"Marker: 0 is not found yet!", (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
	if markerOne is False:
		cv2.putText(frame, f"Marker: 1 is not found yet!", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
	
	default_instruction_texts()
	
	print(marker_size)
	print(f"Total displacement: {total_displacement_in_px}")
	if total_displacement_in_px > 0:
		displacement = (arucoMarkerSizeInFt / np.mean(marker_size)) * total_displacement_in_px
		cv2.putText(frame, "Sinked: {:.2f}ft".format(displacement), (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
		
	# show the output frame
	cv2.imshow("Frame", frame)
	
	# if the `ESC` key was pressed, break from the loop
	if key == 27:
		print(markerZeroCoordinates)
		print("-------------------")
		print(markerOneCoordinates)
		break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
