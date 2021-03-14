import platform
import cv2
from imutils.video import VideoStream

selected_camera_port = 0

if platform.system() == "Windows":
	vs = VideoStream(selected_camera_port + cv2.CAP_DSHOW).start()
else:
	vs = VideoStream(selected_camera_port).start()

while True:
	frame = vs.read()
	
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	
	if key == 27:
		cv2.destroyAllWindows()
		vs.stop()
		break
