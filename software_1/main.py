# import the necessary packages
import dlib
from cffi.backend_ctypes import xrange
from imutils.video import FileVideoStream
from imutils.video import VideoStream
import numpy as np
import time
import cv2

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
tracker = []
detected_corners = []
marker_labels = []
frame_no = 0

# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = FileVideoStream("test_video1.mp4").start()
# vs = VideoStream(0).start()
time.sleep(2.0)


# instruction texts
def default_instruction_texts():
    cv2.putText(frame, "Key r - Restart the program",
                (50, 130), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (250, 0, 20), 2)
    cv2.putText(frame, "Key ESC - Quit the program",
                (50, 180), cv2.FONT_HERSHEY_SIMPLEX,
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

    # reset detection corners
    if len(corners) == 2 and frame_no % 30 == 0 and frame_no > 30:
        detected_corners.clear()
        markerZero = False
        markerOne = False

    # verify *at least* one ArUco marker was detected
    if len(corners) > 0 and (markerZero is False or markerOne is False):
        # flatten the ArUco IDs list
        ids = ids.flatten()
        print(ids)
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
                detected_corners.append((int(topLeft[0]), int(topLeft[1]), int(bottomRight[0]), int(bottomRight[1])))
                marker_labels.append(markerID)
            if markerID == 1 and markerOne is False:
                markerOne = True
                markerOneCoordinates.append(cY)  # save the initial position (Y) of the marker_one
                detected_corners.append((int(topLeft[0]), int(topLeft[1]), int(bottomRight[0]), int(bottomRight[1])))
                marker_labels.append(markerID)

            if markerZero and markerOne:
                # Create the tracker objects
                tracker = [dlib.correlation_tracker() for _ in xrange(len(detected_corners))]
                # print(detected_corners)
                # Provide the tracker the initial position of the object
                [tracker[i].start_track(frame, dlib.rectangle(*rect)) for i, rect in enumerate(detected_corners)]

                # save displacement
                is_saved = save_displacement(markerID, cY)
                save_marker_size(topLeft, topRight, bottomLeft, bottomRight)

            # draw the ArUco marker ID on the frame
            cv2.putText(frame, str(markerID),
                        (topLeft[0], topLeft[1] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 2)

    elif markerZero is True and markerOne is True:
        for i in xrange(len(tracker)):
            tracker[i].update(frame)
            # Get the position of th object, draw a
            # bounding box around it and display it.
            rect = tracker[i].get_position()
            pt1 = (int(rect.left()), int(rect.top()))
            pt2 = (int(rect.right()), int(rect.bottom()))
            cY = int((pt1[1] + pt2[1]) / 2.0)
            is_saved = save_displacement(marker_labels[i], cY)
            if is_saved:
                cv2.rectangle(frame, pt1, pt2, (255, 255, 255), 1)

    default_instruction_texts()
    total_displacement_in_px = measure_displacement()
    print(total_displacement_in_px)
    if total_displacement_in_px > 0:
        displacement = (arucoMarkerSizeInFt / np.mean(marker_size)) * total_displacement_in_px
        cv2.putText(frame, "Sinked: {:.2f}ft".format(displacement), (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),
                    2)
    if markerZero is False:
        cv2.putText(frame, f"Marker: 0 is not found yet!", (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    if markerOne is False:
        cv2.putText(frame, f"Marker: 1 is not found yet!", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    frame_no = frame_no + 1
    # if the `ESC` key was pressed, break from the loop
    if key == 27:
        print(markerZeroCoordinates)
        print("------------------")
        print(markerOneCoordinates)
        break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
