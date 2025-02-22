import cv2
import numpy as np
import toml
import os
import glob
import typer
from config import Config
from typing import List, Tuple

cfg = Config()

COLOR_POS = cfg.COLOR_POS
COLOR_NEG = cfg.COLOR_NEG
GRID_SIZE = cfg.GRID_SIZE

clicked_points: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []
current_label: Tuple[int, int] = (0, 0)
centers: List[Tuple[int, int]] = []
image_with_labels = None
mouse_position: Tuple[int, int] = (0, 0)

app = typer.Typer()

def mouse_callback(event: int, x: int, y: int, flags: int, param):
	global clicked_points, current_label, centers, image_with_labels, mouse_position

	mouse_position = (x, y)

	if event == cv2.EVENT_LBUTTONDOWN:
		#find the closest blob center to the clicked point
		closest_point = None
		min_distance = float('inf')
		for center in centers:
			distance = np.linalg.norm(np.array(center) - np.array([x, y]))
			if distance < min_distance:
				min_distance = distance
				closest_point = tuple(map(int, center))

		if closest_point:
			#append the closest point and its label
			clicked_points.append((closest_point, current_label))
			typer.echo(f"Point at {closest_point}: Label {current_label}")

			#draw the point and label on the image_with_labels
			cv2.circle(image_with_labels, closest_point, 5, (0, 0, 255), -1)
			label = f"({current_label[0]},{current_label[1]})"
			cv2.putText(image_with_labels, label, (closest_point[0] + 10, closest_point[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
			cv2.imshow("Select Points", image_with_labels)

			if current_label[0] < GRID_SIZE[0] - 1:
				current_label = (current_label[0] + 1, current_label[1])
			else:
				current_label = (0, current_label[1] + 1)

def detect_blob_centers(image_path: str, color_mode: str, detect_blobs: bool = True):
	global image, centers, clicked_points, current_label, image_with_labels, mouse_position

	clicked_points = []
	current_label = (0, 0)

	image = cv2.imread(image_path, cv2.IMREAD_COLOR)

	if image is None:
		typer.echo(f"Error: Unable to load image at {image_path}", err=True)
		return []

	#create a mask for COLOR_POS and COLOR_NEG
	lower_pos = np.array(COLOR_POS, dtype=np.uint8)
	upper_pos = np.array(COLOR_POS, dtype=np.uint8)
	lower_neg = np.array(COLOR_NEG, dtype=np.uint8)
	upper_neg = np.array(COLOR_NEG, dtype=np.uint8)

	mask_pos = cv2.inRange(image, lower_pos, upper_pos)
	mask_neg = cv2.inRange(image, lower_neg, upper_neg)

	if color_mode == "both":
		combined_mask = cv2.bitwise_or(mask_pos, mask_neg)
	elif color_mode == "positive":
		combined_mask = mask_pos
	elif color_mode == "negative":
		combined_mask = mask_neg
	else:
		raise ValueError("Invalid color mode. Use 'both', 'positive', or 'negative'.")

	# morphological operations to clean up the mask
	kernel = np.ones((5, 5), np.uint8)
	combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
	combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)

	contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	centers = []

	if detect_blobs:
		for contour in contours:
			M = cv2.moments(contour)
			if M["m00"] != 0:
				cx = int(M["m10"] / M["m00"])
				cy = int(M["m01"] / M["m00"])
				centers.append((cx, cy))

	image_with_blobs = image.copy()
	for center in centers:
		cv2.circle(image_with_blobs, center, 5, (0, 255, 0), -1)  # Green dot for detected blobs

	image_with_labels = image_with_blobs.copy()

	window_name = "Select Points"
	cv2.namedWindow(window_name)
	cv2.setMouseCallback(window_name, mouse_callback)

	cv2.imshow(window_name, image_with_labels)

	#wait for user interaction
	while True:
		key = cv2.waitKey(1) & 0xFF
		if key == ord('n'):  #add a new blob at the current mouse position
			x, y = mouse_position
			centers.append((x, y))
			cv2.circle(image_with_labels, (x, y), 5, (0, 255, 0), -1)  # Green dot for new blob
			cv2.imshow(window_name, image_with_labels)
		elif key == ord('r'):  #reset the labeling
			clicked_points = []
			current_label = (0, 0)
			image_with_labels = image_with_blobs.copy()
			cv2.imshow(window_name, image_with_labels)
		elif key == ord('c'):  #move to the next image
			break
		elif key == 27:  # ESC key to exit
			cv2.destroyAllWindows()
			return None  # Exit the program

	cv2.destroyAllWindows()

	return clicked_points

def save_labeled_data(image_paths: List[str], labeled_points: List[List[Tuple[Tuple[int, int], Tuple[int, int]]]], output_file: str):
	labeled_data = {"images": []}

	for image_path, points in zip(image_paths, labeled_points):
		image_data = {
			"path": image_path,
			"points": [
				{
					"image_coordinates": {"x": point[0], "y": point[1]},
					"world_coordinates": {"x": label[0], "y": label[1]}
				}
				for point, label in points
			]
		}
		labeled_data["images"].append(image_data)

	with open(output_file, "a") as f:
		toml.dump(labeled_data, f)
	typer.echo(f"Labeled points saved to '{output_file}'.")

@app.command()
def main(
	folder: str = typer.Argument(..., help="Path to the folder containing images."),
	output: str = typer.Argument(..., help="Output TOML file name."),
	color_mode: str = typer.Option(
		"both",
		"--color",
		"-c",
		help="Color mode for blob detection: 'both', 'positive', or 'negative'.",
		case_sensitive=False,
	),
	detect_blobs: bool = typer.Option(
		True,
		"--no-detect",
		help="Detect blobs in images.",
	),
):
	"""
	Detect blob centers in images and save labeled points to a TOML file.

	Commands:
	  n: Add a new blob at the current mouse position.
	  r: Reset the labeling for the current image.
	  c: Move to the next image.
	  ESC: Exit the program.
	"""

	if color_mode not in ["both", "positive", "negative"]:
		typer.echo("Invalid color mode. Use 'both', 'positive', or 'negative'.", err=True)
		raise typer.Exit(1)

	image_paths = glob.glob(os.path.join(folder, "*.png"))

	for idx, image_path in enumerate(image_paths):
		typer.echo(f"Processing image {idx + 1}/{len(image_paths)}: {image_path}")
		points = detect_blob_centers(image_path, color_mode, detect_blobs)
		if points is None:  # Exit if ESC was pressed
			break
		#labeled_points.append(points)

		if points:  # save data only if labeling was completed
			save_labeled_data([image_path, ], [points, ], output)

if __name__ == "__main__":
	app()