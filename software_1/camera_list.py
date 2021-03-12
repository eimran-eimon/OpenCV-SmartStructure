import cv2
import platform


def working_camera_list_ports():
	is_working = True
	dev_port = 0
	working_ports = []
	available_ports = []
	while is_working:
		if platform.system() == "Windows":
			camera = cv2.VideoCapture(dev_port + cv2.CAP_DSHOW)
		else:
			camera = cv2.VideoCapture(dev_port)

		if not camera.isOpened():
			is_working = False
		else:
			is_reading, img = camera.read()
			w = camera.get(3)
			h = camera.get(4)
			if is_reading:
				print("Port %s is working and reads images (%s x %s)" % (dev_port, h, w))
				working_ports.append(dev_port)
			else:
				print("Port %s for camera ( %s x %s) is present but does not reads." % (dev_port, h, w))
				available_ports.append(dev_port)

		camera.release()
		dev_port += 1

	return working_ports