# import the necessary packages
from imutils.video import FileVideoStream
from imutils.video import VideoStream
import numpy as np
import platform
import cv2
import gui


def automated_mode_run():
	
	arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
	arucoParams = cv2.aruco.DetectorParameters_create()
	arucoParams.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
	arucoMarkerSizeInFt = 0.575  # 6.9 inch
	
	markerZero = False  # is marker zero detected
	markerOne = False  # is marker one detected
	
	markerZeroCoordinates = []  # save the path of the marker zero
	markerOneCoordinates = []  # save the path of the marker one
	
	# displacement in pixel
	total_displacement_in_px = 0
	# calculate and save marker size as long as program run to get more accurate data
	marker_size = []
	
	# GUI
	input_method = gui.browse_or_camera()
	if input_method == 0:
		print("[INFO] starting video stream...")
		selected_camera_port = gui.show_list_of_cameras()
		if platform.system() == "Windows":
			vs = VideoStream(selected_camera_port + cv2.CAP_DSHOW).start()
		else:
			vs = VideoStream(selected_camera_port).start()

	elif input_method == 1:
		print("[INFO] starting file video stream...")
		file_name = gui.browse_sample_video()
		vs = FileVideoStream(file_name).start()
	
	# instruction texts
	def default_instruction_texts():
		# cv2.putText(frame, "Key r - Restart the program",
		#             (50, 130), cv2.FONT_HERSHEY_SIMPLEX,
		#             0.7, (250, 0, 20), 2)
		cv2.putText(frame, "Key ESC - Quit the program",
		            (50, 130), cv2.FONT_HERSHEY_SIMPLEX,
		            0.7, (250, 0, 20), 2)
	
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
			prev_y = markerZeroCoordinates[-1]
			if current_y - prev_y > 0:
				markerZeroCoordinates.append(current_y)
				return True
		elif detected_marker_id == 1:
			prev_y = markerOneCoordinates[-1]
			if current_y - prev_y > 0:
				markerOneCoordinates.append(current_y)
				return True
		else:
			return False
	
	# loop over the frames from the video stream
	def save_marker_size(top_left, top_right, bottom_left, bottom_right):
		marker_size.append(np.abs(top_left[0] - top_right[0]))
		marker_size.append(np.abs(top_right[1] - bottom_right[1]))
		marker_size.append(np.abs(bottom_right[0] - bottom_left[0]))
		marker_size.append(np.abs(bottom_left[1] - top_left[1]))
	
	while True:
		# grab the frame from the threaded video stream
		# and resize it to have a maximum width of 1000 pixels
		frame = vs.read()
		# frame = imutils.resize(frame, width=1000)
		
		if frame is None:
			break
		
		# detect ArUco markers in the input frame
		(corners, ids, rejected) = cv2.aruco.detectMarkers(frame, arucoDict, parameters=arucoParams)
		
		# verify *at least* one ArUco marker was detected
		if len(corners) > 0:
			# flatten the ArUco IDs list
			ids = ids.flatten()
			
			# loop over the detected ArUCo corners
			for (markerCorner, markerID) in zip(corners, ids):
				# extract the marker corners (which are always returned
				# in top-left, top-right, bottom-right, and bottom-left
				# order)
				corners = markerCorner.reshape((4, 2))
				(topLeft, topRight, bottomRight, bottomLeft) = corners
				
				# convert each of the (x, y)-coordinate pairs to integers
				topRight = (int(topRight[0]), int(topRight[1]))
				bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
				bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
				topLeft = (int(topLeft[0]), int(topLeft[1]))
				
				# draw the bounding box of the ArUCo detection
				cv2.line(frame, topLeft, topRight, (0, 255, 0), 2)
				cv2.line(frame, topRight, bottomRight, (0, 255, 0), 2)
				cv2.line(frame, bottomRight, bottomLeft, (0, 255, 0), 2)
				cv2.line(frame, bottomLeft, topLeft, (0, 255, 0), 2)
				
				# compute and draw the center (x, y)-coordinates of the
				# ArUco marker
				cX = int((topLeft[0] + bottomRight[0]) / 2.0)
				cY = int((topLeft[1] + bottomRight[1]) / 2.0)
				cv2.circle(frame, (cX, cY), 4, (0, 0, 255), -1)
				
				if markerID == 0 and markerZero is False:
					markerZero = True
					markerZeroCoordinates.append(cY)  # save the initial position (Y) of the marker_zero
				if markerID == 1 and markerOne is False:
					markerOne = True
					markerOneCoordinates.append(cY)  # save the initial position (Y) of the marker_one
				
				if markerZero and markerOne:
					is_saved = save_displacement(markerID, cY)
					if is_saved is True:
						save_marker_size(topLeft, topRight, bottomLeft, bottomRight)
						if len(ids) == 1:
							# if one marker is missing
							copy_displacement_of_the_other_marker(markerID)
					total_displacement_in_px = measure_displacement()
				
				# draw the ArUco marker ID on the frame
				cv2.putText(frame, str(markerID), (topLeft[0], topLeft[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
				            (0, 255, 0), 2)
		
		default_instruction_texts()
		# print(np.mean(marker_size))
		# print(total_displacement_in_px)
		if total_displacement_in_px > 0:
			displacement = (arucoMarkerSizeInFt / np.mean(marker_size)) * total_displacement_in_px
			cv2.putText(frame, "Sinked: {:.2f}ft".format(displacement), (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
			            (0, 0, 255), 2)
		if markerZero is False:
			cv2.putText(frame, f"Marker: 0 is not found yet!", (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
		if markerOne is False:
			cv2.putText(frame, f"Marker: 1 is not found yet!", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
		
		# show the output frame
		cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF
		
		# if the `ESC` key was pressed, break from the loop
		if key == 27:
			cv2.destroyAllWindows()
			vs.stop()
			break
	
	# do a bit of cleanup
	cv2.destroyAllWindows()
	vs.stop()
