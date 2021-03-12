import sys
import numpy as np
import cv2
import manual_mode
import auto_mode
import platform

while True:
	frame = np.zeros((500, 500, 3))
	
	cv2.putText(frame, "*Press the respective key",
				(80, 80), cv2.FONT_HERSHEY_SIMPLEX,
				0.5, (250, 200, 20), 1)
	
	cv2.putText(frame, "Key a - Automated Mode",
				(80, 130), cv2.FONT_HERSHEY_SIMPLEX,
				0.7, (250, 0, 20), 2)
	cv2.putText(frame, "Key m - Manual Mode",
				(80, 180), cv2.FONT_HERSHEY_SIMPLEX,
				0.7, (250, 0, 20), 2)
	cv2.putText(frame, "Key r - Restart the program",
	            (80, 230), cv2.FONT_HERSHEY_SIMPLEX,
	            0.7, (250, 0, 20), 2)
	
	cv2.imshow('Mode Selection', frame)
	k = cv2.waitKey(1)
	
	if k == ord('a'):
		auto_mode.automated_mode_run()
	if k == ord('m'):
		manual_mode.manual_mode_run()
	if k == ord('r'):
		cv2.destroyAllWindows()
		cv2.os.execl(sys.executable, sys.executable, *sys.argv)
		
	if k == 27:
		# release resources
		cv2.destroyAllWindows()
		break
