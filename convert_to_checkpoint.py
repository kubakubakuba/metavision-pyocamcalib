import toml
import pickle
import numpy as np
import typer

def convert_toml_to_checkpoint(toml_file: str, checkpoint_file: str, prefix: str = ""):
	with open(toml_file, "r") as f:
		toml_data = toml.load(f)

	checkpoint_data = {}

	for image_data in toml_data["images"]:
		image_path = prefix + image_data["path"]  #add prefix to the file path
		image_points = []
		world_points = []

		for point in image_data["points"]:
			image_points.append([point["image_coordinates"]["x"], point["image_coordinates"]["y"]])
			world_points.append([point["world_coordinates"]["x"], point["world_coordinates"]["y"], 0.0])  # Add z=0

		#convert to numpy arrays
		image_points = np.array(image_points, dtype=np.float32)
		world_points = np.array(world_points, dtype=np.float32)

		checkpoint_data[image_path] = {
			"image_points": image_points,
			"world_points": world_points,
		}

	#save the checkpoint data as a pickle file, this can be read by pyocamcalib
	with open(checkpoint_file, "wb") as f:
		pickle.dump(checkpoint_data, f)

	typer.echo(f"Checkpoint saved to {checkpoint_file}")

def main(toml_file: str, checkpoint_file: str, prefix: str = ""):
	"""
	Main function to convert TOML data to checkpoint format.

	Args:\n
		toml_file (str): Path to the input TOML file.\n
		checkpoint_file (str): Path to the output checkpoint file.\n
		prefix (str): Optional prefix to prepend to file paths.
	"""
	convert_toml_to_checkpoint(toml_file, checkpoint_file, prefix)

if __name__ == "__main__":
	typer.run(main)