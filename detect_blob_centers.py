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
POINT_SIZE = cfg.POINT_SIZE
POINT_COLOR = cfg.POINT_COLOR
TXT_SIZE = cfg.TXT_SIZE
TXT_COLOR = cfg.TXT_COLOR
TXT_OFFSET = cfg.TXT_OFFSET
POINT_CLICKED_COLOR = cfg.POINT_CLICKED_COLOR

clicked_points: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []
current_label: Tuple[int, int] = (0, 0)
centers: List[Tuple[int, int]] = []
image_with_labels = None
mouse_position: Tuple[int, int] = (0, 0)
original_image = None

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
			cv2.circle(image_with_labels, closest_point, POINT_SIZE, POINT_CLICKED_COLOR, -1)
			label = f"({current_label[0]},{current_label[1]})"
			cv2.putText(image_with_labels, label, (closest_point[0] + TXT_OFFSET[0], closest_point[1] + TXT_OFFSET[1]), cv2.FONT_HERSHEY_SIMPLEX, TXT_SIZE, TXT_COLOR, 2)
			cv2.imshow("Select Points", image_with_labels)

			if current_label[0] < GRID_SIZE[0] - 1:
				current_label = (current_label[0] + 1, current_label[1])
			else:
				current_label = (0, current_label[1] + 1)

def detect_blobs(image: np.ndarray) -> List[Tuple[int, int]]:
	"""Detect blob centers in the given image"""
	blurred_image = cv2.GaussianBlur(image, (5, 5), 0)

	_, binary_mask = cv2.threshold(blurred_image, 50, 255, cv2.THRESH_BINARY)

	contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	centers = []
	for contour in contours:
		mask = np.zeros_like(image)
		cv2.drawContours(mask, [contour], -1, 255, -1)

		masked_image = cv2.bitwise_and(image, image, mask=mask)
		_, max_val, _, max_loc = cv2.minMaxLoc(masked_image)

		if max_val > 0:
			centers.append(max_loc)
	
	return centers

def detect_blob_centers(image_path: str, color_mode: str, detect_blobs_flag: bool = True):
	global image, centers, clicked_points, current_label, image_with_labels, mouse_position, original_image

	clicked_points = []
	current_label = (0, 0)

	image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
	original_image = image.copy()

	if image is None:
		typer.echo(f"Error: Unable to load image at {image_path}", err=True)
		return []

	if detect_blobs_flag:
		centers = detect_blobs(image)
	else:
		centers = []

	image_with_blobs = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
	for center in centers:
		cv2.circle(image_with_blobs, center, POINT_SIZE, POINT_COLOR, -1)  # green dot for detected blobs

	image_with_labels = image_with_blobs.copy()

	window_name = "Select Points"
	cv2.namedWindow(window_name)
	cv2.setMouseCallback(window_name, mouse_callback)

	cv2.imshow(window_name, image_with_labels)

	while True:
		key = cv2.waitKey(1) & 0xFF
		if key == ord('n'):
			x, y = mouse_position
			centers.append((x, y))
			cv2.circle(image_with_labels, (x, y), POINT_SIZE, POINT_COLOR, -1)
			cv2.imshow(window_name, image_with_labels)

		elif key == ord('x'): #remove closest detected blob center
			if centers:
				closest_idx = None
				min_distance = float('inf')
				for i, center in enumerate(centers):
					distance = np.linalg.norm(np.array(center) - np.array(mouse_position))
					if distance < min_distance:
						min_distance = distance
						closest_idx = i
				
				if closest_idx is not None:
					removed_center = centers.pop(closest_idx)
					
					clicked_points = [pt for pt in clicked_points if pt[0] != removed_center]
					
					image_with_labels = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
					for center in centers:
						cv2.circle(image_with_labels, center, POINT_SIZE, POINT_COLOR, -1)
					
					for point, label in clicked_points:
						cv2.circle(image_with_labels, point, POINT_SIZE, POINT_CLICKED_COLOR, -1)
						label_text = f"({label[0]},{label[1]})"
						cv2.putText(image_with_labels, label_text, 
									(point[0] + TXT_OFFSET[0], point[1] + TXT_OFFSET[1]), 
									cv2.FONT_HERSHEY_SIMPLEX, TXT_SIZE, TXT_COLOR, 2)
					
					cv2.imshow(window_name, image_with_labels)
					typer.echo(f"Removed point at {removed_center}")

		elif key == ord('r'): #reset the labeling
			clicked_points = []
			current_label = (0, 0)
			image = original_image.copy()
			
			if detect_blobs_flag:
				centers = detect_blobs(image)
			else:
				centers = []
			
			# Recreate the display image
			image_with_labels = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
			for center in centers:
				cv2.circle(image_with_labels, center, POINT_SIZE, POINT_COLOR, -1)
			
			cv2.imshow(window_name, image_with_labels)
			typer.echo("Reset complete - re-detected blob centers")
		elif key == ord('c'):  # move to the next image
			break
		elif key == 27:  # ESC key to exit
			cv2.destroyAllWindows()
			return None

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
	  x: Remove the closest blob to the mouse position.
	  r: Reset the labeling and re-detect blob centers.
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

		if points:  # save data only if labeling was completed
			save_labeled_data([image_path], [points], output)

if __name__ == "__main__":
	app()