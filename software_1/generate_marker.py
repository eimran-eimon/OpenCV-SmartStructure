import cv2
import numpy as np
from fpdf import FPDF

pdf = FPDF('P', 'mm', 'A4')
pdf.set_font('Arial', 'B', 16)
no_of_markers = 2

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
tag = np.zeros((500, 500, 1), dtype="uint8")
for i in range(0, no_of_markers):
	cv2.aruco.drawMarker(dictionary, 1, 500, tag, 1)
	cv2.imwrite(f"marker_{i}.png", tag)
	pdf.add_page()
	pdf.cell(20, 20, f"Marker_{i}")
	pdf.image(f"marker_{i}.png", x=20, y=40)

pdf.output("markers.pdf", "F")
