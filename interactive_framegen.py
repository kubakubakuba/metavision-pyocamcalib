import cv2
import os
import random
import string
import typer
from config import Config
from generate_frames import accumulate_events

cfg = Config()

FAST_STEP = cfg.FAST_STEP  # Step size for 'j' and 'l' keys
SLOW_STEP = cfg.SLOW_STEP  # Step size for 'a' and 'd' keys
FINE_STEP = cfg.FINE_STEP  # Step size for 'b' and 'm' keys
SUPERFINE_STEP = cfg.SUPERFINE_STEP  # Step size for 'n' and 'h' keys

# Initialize Typer app
app = typer.Typer()

def random_hex_string(length: int = 8) -> str:
	"""Generate a random hexadecimal string of a given length."""
	return ''.join(random.choice(string.hexdigits) for _ in range(length))

def format_time_us(time_us: int) -> str:
	"""Convert time in microseconds to a formatted string in minutes:seconds.decimal."""
	seconds = time_us / 1_000_000
	minutes = int(seconds // 60)
	remaining_seconds = seconds % 60
	return f"{minutes}:{remaining_seconds:.6f}"

@app.command()
def main(
	input_file: str = typer.Argument(..., help="Path to the input raw file."),
	output_folder: str = typer.Argument(..., help="Folder to save the frames."),
	start_time_us: int = typer.Option(0, help="Start time in microseconds."),
	accumulation_time_us: int = typer.Option(10000, help="Accumulation time in microseconds."),
	threshold: int = typer.Option(1, help="Threshold for event accumulation to filter noise."),
):
	"""
	Generate frames from a raw recording interactively.\n\n

	Use the following keys for navigation:\n
	  'g' - Generate frame\n
	  's' - Save frame to the output folder\n
	  'j' - Fast rewind\n
	  'l' - Fast forward\n
	  'a' - Slow rewind\n
	  'd' - Slow forward\n
	  'b' - Superfine rewind\n
	  'm' - Superfine forward\n
	  'h' - Increase accumulation time\n
	  'n' - Decrease accumulation time\n
	  'q' - Quit
	"""
	current_time_us = start_time_us
	frame = None

	os.makedirs(output_folder, exist_ok=True)

	cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)

	#display the initial logo
	frame = cv2.imread('./docs/controls.png')
	if frame is not None:
		cv2.imshow('Frame', frame)

	while True:
		key = cv2.waitKey(1) & 0xFF  # small delay for responsiveness

		if key == ord('q'):  # Quit
			break
		elif key == ord('g'):  # Generate frame
			typer.echo("Generating frame...")
			output_file = os.path.join(output_folder, "temp_frame.png")
			accumulate_events(input_file, current_time_us, accumulation_time_us, output_file, threshold)
			frame = cv2.imread(output_file)

			if frame is None:
				typer.echo("No more frames to process.", err=True)
				break

			cv2.imshow('Frame', frame)
		elif key == ord('s'):  # Save frame
			if frame is not None:
				filename = f"frame_{random_hex_string()}.png"
				filepath = os.path.join(output_folder, filename)
				cv2.imwrite(filepath, frame)
				typer.echo(f"Saved frame to {filepath}")
			else:
				typer.echo("No frame to save. Generate a frame first.", err=True)
		elif key == ord('j'):  # Fast rewind
			current_time_us -= FAST_STEP
			typer.echo(f"Current time: {format_time_us(current_time_us)}")
		elif key == ord('l'):  # Fast forward
			current_time_us += FAST_STEP
			typer.echo(f"Current time: {format_time_us(current_time_us)}")
		elif key == ord('a'):  # Slow rewind
			current_time_us -= SLOW_STEP
			typer.echo(f"Current time: {format_time_us(current_time_us)}")
		elif key == ord('d'):  # Slow forward
			current_time_us += SLOW_STEP
			typer.echo(f"Current time: {format_time_us(current_time_us)}")
		elif key == ord('b'):  # Superfine rewind
			current_time_us -= FINE_STEP
			typer.echo(f"Current time: {format_time_us(current_time_us)}")
		elif key == ord('m'):  # Superfine forward
			current_time_us += FINE_STEP
			typer.echo(f"Current time: {format_time_us(current_time_us)}")
		elif key == ord('h'):  # Increase accumulation time
			accumulation_time_us += SUPERFINE_STEP
			typer.echo(f"Current acc time: {accumulation_time_us}")
		elif key == ord('n'):  # Decrease accumulation time
			accumulation_time_us -= SUPERFINE_STEP
			typer.echo(f"Current acc time: {accumulation_time_us}")
		elif key == ord('i'):  # Increase threshold
			threshold += 1
			typer.echo(f"Threshold: {threshold}")
		elif key == ord('k'):  # Decrease threshold
			threshold = max(1, threshold - 1)
			typer.echo(f"Threshold: {threshold}")

		# Ensure time doesn't go negative
		current_time_us = max(0, current_time_us)

	cv2.destroyAllWindows()

if __name__ == "__main__":
	app()