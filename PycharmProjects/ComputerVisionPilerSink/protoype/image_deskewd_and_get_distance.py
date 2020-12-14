import cv2
import numpy as np
from scipy.ndimage import interpolation as inter
import matplotlib.pyplot as plt
import math

row_1 = plt.subplot(1, 3, 1)
row_2 = plt.subplot(1, 3, 2)
row_3 = plt.subplot(1, 3, 3)


def skew_correction(image, delta=1, limit=10):
	def determine_score(arr, angle):
		projected_data = inter.rotate(arr, angle, reshape=False, order=0)
		histogram = np.sum(projected_data, axis=1)
		score = np.sum((histogram[1:] - histogram[:-1]) ** 2)
		return histogram, score

	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
	thresh = cv2.erode(thresh, np.ones((1, 5), np.uint8))
	thresh = cv2.dilate(thresh, np.ones((1, 5), np.uint8))
	scores = []
	stored_histogram = []
	angles = np.arange(-limit, limit + delta, delta)
	for angle in angles:
		histogram, score = determine_score(thresh, angle)
		scores.append(score)
		stored_histogram.append(histogram)

	best_angle = angles[scores.index(max(scores))]
	best_histogram = stored_histogram[scores.index(max(scores))]

	(h, w) = image.shape[:2]
	center = (w // 2, h // 2)
	M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
	rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

	return best_angle, rotated, best_histogram


def zero_runs(a):
	# Create an array that is 1 where a is 0, and pad each end with an extra 0.
	iszero = np.concatenate(([0], np.equal(a, 0).view(np.int8), [0]))
	absdiff = np.abs(np.diff(iszero))
	# Runs start and end where absdiff is 1.
	ranges = np.where(absdiff == 1)[0].reshape(-1, 2)

	return ranges


def get_distance(histogram):
	ranges = zero_runs(histogram)
	differences = np.diff(ranges)
	remove_noisy_diff = [y1 for y1 in differences if (15 <= y1 <= 25)]
	return np.mean(remove_noisy_diff)





def draw_plots(org_img, rotated_img, angle, histogram):
	edge_heatmap = None
	edge_heatmap = cv2.normalize(histogram, edge_heatmap, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
	edge_heatmap = cv2.applyColorMap(edge_heatmap, cv2.COLORMAP_JET)
	resized_heatmap = cv2.resize(edge_heatmap, (org_img.shape[1], org_img.shape[0]), interpolation=cv2.INTER_CUBIC)
	cv2.imwrite(f"best_heatmap.png", resized_heatmap)

	row_1.text(-20, -15, 'Original Image', bbox={'facecolor': 'white', 'pad': 10})
	row_1.imshow(cv2.cvtColor(org_img, cv2.COLOR_BGR2RGB))

	row_2.text(-20, -15, f'De-skewed by {angle}°', bbox={'facecolor': 'white', 'pad': 10})
	row_2.imshow(cv2.cvtColor(rotated_img, cv2.COLOR_BGR2RGB))

	return resized_heatmap


if __name__ == '__main__':
	image = cv2.imread('t3.png')
	final_angle, rotated_image, max_scored_histogram = skew_correction(image)
	heatmap = draw_plots(image, rotated_image, final_angle, max_scored_histogram)
	distance = get_distance(max_scored_histogram)
	if not math.isnan(distance) and distance > 0:
		row_3.text(-20, -15, f'Heatmap, 1ft = {distance}px', bbox={'facecolor': 'white', 'pad': 10})
		row_3.imshow(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB))
	plt.show()
	# if 'ESC' is pressed, quit the program
	k = cv2.waitKey(30) & 0xff
	if k == 27:
		# release resources
		cv2.destroyAllWindows()
