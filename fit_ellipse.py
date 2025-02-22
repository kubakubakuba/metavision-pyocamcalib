import matplotlib
matplotlib.use('tkagg')

from matplotlib import pyplot as plt
from metavision_core.event_io import RawReader
import numpy as np

from scipy.optimize import minimize
from matplotlib.patches import Ellipse

########### MODIFY THIS SECTION ############

calib_file = 'raws/fish/Entaniya_Fisheye_280_calibration_2025-02-06.raw'
initial_guess = [425, 430]  # [a, b]

# rotation angle (fixed)
theta_opt = 0

############################################


raw = RawReader(calib_file)

h, w = raw.get_size()
print(f"Height: {h}, Width: {w}")
start_ts = 0.5 * 1e6
dt = 7.5 * 1e6

height, width = raw.get_size()
raw.seek_time(start_ts)

delta_t = dt
events = raw.load_delta_t(delta_t)
events['t'] -= int(start_ts)

#convert x and y to int32 immediately after loading to prevent uint16 overflow
original_dtype = events.dtype

new_dtype = []
for field in original_dtype.names:
	if field == 'x' or field == 'y':
		new_dtype.append((field, np.int32))
	else:
		new_dtype.append((field, original_dtype[field]))
events = events.astype(new_dtype)

events_err = events[(events['x'] >= w) | (events['y'] >= h)]

events['x'] -= w // 2
events['y'] -= h // 2
	
hist, x_edges, y_edges = np.histogram2d(
	events['x'],  # x-coordinates
	events['y'],  # y-coordinates
	bins=[w, h],  # number of bins (width and height of the sensor)
	range=[[-w//2, w//2], [-h//2, h//2]]  # range of x and y coordinates
)

plt.figure(figsize=(10, 8))
plt.imshow(np.log1p(hist.T), origin='lower', cmap='viridis', extent=[-w//2, w//2, -h//2, h//2])
plt.colorbar(label='Event Count')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('Event Density Histogram')
plt.show()

x = events['x']
y = events['y']
points = np.column_stack((x, y))

distances = np.sqrt(x**2 + y**2)
percentile_cutoff = np.percentile(distances, 99)
filtered_points = points[distances <= percentile_cutoff]
x_filt, y_filt = filtered_points[:, 0], filtered_points[:, 1]

def ellipse_cost(params, points):
	a, b = params
	x, y = points[:, 0], points[:, 1]
	
	# check if points lie inside the ellipse
	inside = (x**2 / a**2) + (y**2 / b**2) <= 1
	
	# cost: sum of squared distances for points outside the ellipse
	cost = np.sum((x[~inside]**2 / a**2) + (y[~inside]**2 / b**2) - 1)
	return cost


# optimize the ellipse parameters (a and b only)
result = minimize(ellipse_cost, initial_guess, args=(filtered_points,), method='L-BFGS-B')
a_opt, b_opt = result.x

print(f"Optimized Major Axis (a): {a_opt}")
print(f"Optimized Minor Axis (b): {b_opt}")
print(f"Fixed Rotation Angle (theta): {np.degrees(theta_opt)} degrees")


fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(x_filt, y_filt, 'b.', markersize=1, label='Filtered Events')

ellipse = Ellipse(
	xy=(0, 0),
	width=2 * a_opt,
	height=2 * b_opt,
	angle=np.degrees(theta_opt),
	edgecolor='red',
	facecolor='none',
	linewidth=2,
	label='Fitted Ellipse'
)
ax.add_patch(ellipse)

# Axis labels and limits
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('Optimal Encompassing Ellipse (Fixed Rotation)')
plt.legend(loc='upper right')
plt.xlim(-700, 700)
plt.ylim(-700, 700)
plt.grid(True)
plt.show()