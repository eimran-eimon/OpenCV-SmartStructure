# import the necessary packages
import dlib
from imutils.video import FileVideoStream
from imutils.video import VideoStream
import numpy as np
import gui
import cv2
import platform


def manual_mode_run():
	pts = []  # for storing points
	markerZero = False  # is marker zero detected
	markerOne = False  # is marker one detected
	arucoMarkerSizeInFt = 0.974409449  # 11.7 inch
	
	markerZeroCoordinates = []  # save the path of the marker zero
	markerOneCoordinates = []  # save the path of the marker one
	tracker = []
	# displacement in pixel
	total_displacement_in_px = 0
	# calculate and save marker size as long as program run to get more accurate data
	marker_size = []
	marker_labels = []
	
	# GUI
	input_method = gui.browse_or_camera()
	if input_method == 0:
		print("[INFO] starting video stream...")
		selected_camera_port = gui.show_list_of_cameras()
		if platform.system() == "Windows":
			vs = VideoStream(selected_camera_port + + cv2.CAP_DSHOW).start()
		else:
			vs = VideoStream(selected_camera_port).start()
	# time.sleep(2.0)
	elif input_method == 1:
		print("[INFO] starting file video stream...")
		file_name = gui.browse_sample_video()
		vs = FileVideoStream(file_name).start()
	
	def draw_roi(event, x, y, flags, param):
		img = frame.copy()
		
		if event == cv2.EVENT_LBUTTONDOWN:
			pts.append((x, y))
		
		if event == cv2.EVENT_RBUTTONDOWN:  # Right click to cancel the last selected point
			pts.pop()
		
		if event == cv2.EVENT_MBUTTONDOWN:
			mask = np.zeros(img.shape, np.uint8)
			points = np.array(pts, np.int32)
			points = points.reshape((-1, 1, 2))
	
			mask = cv2.polylines(mask, [points], True, (255, 255, 255), 2)
			mask2 = cv2.fillPoly(mask.copy(), [points], (255, 255, 255))  # for ROI
			mask3 = cv2.fillPoly(mask.copy(), [points], (0, 255, 0))  # for displaying images on the desktop
	
			show_image = cv2.addWeighted(src1=img, alpha=0.8, src2=mask3, beta=0.2, gamma=0)
			
			cv2.imshow("mask", mask2)
			cv2.imshow("show_img", show_image)
	
			ROI = cv2.bitwise_and(mask2, img)
			cv2.imshow("ROI", ROI)
			cv2.waitKey(0)
		
		if len(pts) > 0:
			# Draw the last point in pts
			cv2.circle(img, pts[-1], 3, (0, 0, 255), -1)
		
		if len(pts) > 1:
			for i in range(len(pts) - 1):
				cv2.circle(img, pts[i], 5, (0, 0, 255), -1)  # x ,y is the coordinates of the mouse click place
			cv2.line(img=img, pt1=pts[i], pt2=pts[i + 1], color=(255, 0, 0), thickness=2)
		
		cv2.imshow('Frame', img)
		
	# instruction texts
	def default_instruction_texts():
		# cv2.putText(frame, "Key c - Cancel the selections", (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (250, 0, 20), 2)
		cv2.putText(frame, "Key ESC - Quit the program", (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (250, 0, 20), 2)
	
	def measure_displacement():
		if len(markerZeroCoordinates) < 2 or len(markerOneCoordinates) < 2:
			return 0
		marker_zero_total_displacement = markerZeroCoordinates[-1] - markerZeroCoordinates[0]
		marker_one_total_displacement = markerOneCoordinates[-1] - markerOneCoordinates[0]
		total_displacement = np.mean([marker_zero_total_displacement, marker_one_total_displacement])
		return total_displacement
	
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
		
		cv2.namedWindow('Frame')
		cv2.setMouseCallback('Frame', draw_roi)
		
		if key == ord("0") and markerZero is False:
			markerZero = True
			box = cv2.selectROI("Frame", frame, fromCenter=False, showCrosshair=False)
			t = dlib.correlation_tracker()
			rect = dlib.rectangle(int(box[0]), int(box[1]), int(box[0]) + int(box[2]), int(box[1]) + int(box[3]))
			# markerZeroCoordinates.append((int(box[1]) + (int(box[1]) + int(box[3]))) / 2)
			marker_size.append(int(box[3]))
			t.start_track(frame, rect)
			tracker.append(t)
			marker_labels.append(0)
		
		if key == ord("1") and markerOne is False:
			markerOne = True
			box = cv2.selectROI("Frame", frame, fromCenter=False, showCrosshair=False)
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
				if True:  # changed it to is_saved later
					# draw the bounding box from the correlation object tracker
					cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
					cv2.putText(frame, str(l), (startX, startY - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
		
		if markerZero is False:
			cv2.putText(frame, f"key 0 => Select an A4 paper and then press ENTER", (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
		if markerOne is False:
			cv2.putText(frame, f"key 1 => Select an A4 paper and then press ENTER", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
		
		default_instruction_texts()
		
		# print(marker_size)
		# print(f"Total displacement: {total_displacement_in_px}")
		if total_displacement_in_px > 0:
			displacement = (arucoMarkerSizeInFt / np.mean(marker_size)) * total_displacement_in_px
			cv2.putText(frame, "Sinked: {:.2f}ft".format(displacement), (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
		
		# show the output frame
		cv2.imshow("Frame", frame)

		# if the `ESC` key was pressed, break from the loop
		if key == 27:
			cv2.destroyAllWindows()
			vs.stop()
			break
	
	# do a bit of cleanup
	cv2.destroyAllWindows()
	vs.stop()
