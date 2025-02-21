import argparse
import numpy as np
import cv2
from config import Config
from metavision_core.event_io import RawReader

cfg = Config()

COLOR_POS = cfg.COLOR_POS
COLOR_NEG = cfg.COLOR_NEG

def accumulate_events(raw_file, start_time_us, accumulation_time_us, output_file, threshold):
	reader = RawReader(raw_file)

	reader.seek_time(start_time_us)

	height, width = reader.get_size()
	pos_accumulator = np.zeros((height, width), dtype=np.uint16)
	neg_accumulator = np.zeros((height, width), dtype=np.uint16)

	#process events for the specified accumulation time
	end_time_us = start_time_us + accumulation_time_us
	while reader.current_time < end_time_us:
		events = reader.load_delta_t(1000)  #load events in chunks of 1000 microseconds
		if events is None:
			break

		#accumulate events
		for event in events:
			x, y, p, _ = event
			if p > 0:
				pos_accumulator[y, x] += 1
			else:
				neg_accumulator[y, x] += 1

	pos_mask = pos_accumulator >= threshold
	neg_mask = neg_accumulator >= threshold

	#create a color image to visualize positive and negative events
	color_image = np.zeros((height, width, 3), dtype=np.uint8)
	for y in range(height):
		for x in range(width):
			if pos_accumulator[y, x] >= threshold or neg_accumulator[y, x] >= threshold:
				if pos_accumulator[y, x] > neg_accumulator[y, x]:
					color_image[y, x] = COLOR_POS
				else:
					color_image[y, x] = COLOR_NEG

	#save the accumulated image as a PNG file
	cv2.imwrite(output_file, color_image)
	print(f"Accumulated events saved to {output_file}")

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Accumulate events from a raw recording and save as a PNG image.")
	parser.add_argument("--input", type=str, required=True, help="Path to the input raw file.")
	parser.add_argument("--start-time", type=int, required=True, help="Start time in microseconds.")
	parser.add_argument("--accumulation-time", type=int, required=True, help="Accumulation time in microseconds.")
	parser.add_argument("--output", type=str, required=True, help="Output PNG file name.")
	parser.add_argument("--threshold", type=int, default=1, help="Threshold for event accumulation to filter noise.")

	args = parser.parse_args()

	accumulate_events(args.input, args.start_time, args.accumulation_time, args.output, args.threshold)