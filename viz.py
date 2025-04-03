from pyocamcalib.modelling.camera import Camera
import numpy as np
import matplotlib.pyplot as plt

######## MODIFY THIS SECTION ########

#json_file = 'calibration_fish2.json'
json_file = 'calibration_new.json'
# ellipse parameters
#a = 425 #major axis
#b = 430 #minor axis
a, b = (652, 650)
theta = np.radians(0)  #rotation angle

####################################

cam = Camera().load_parameters_json(json_file)

height, width = 720, 1280

y, x = np.mgrid[0:height, 0:width]
uv_points = np.vstack((x.flatten(), y.flatten())).T

uv_points = uv_points.astype(np.float64)
unit_vectors = cam.cam2world(uv_points)

unit_vectors = unit_vectors.reshape(height, width, 3)

optical_axis = np.array([0, 0, 1])

angles = np.arccos(np.dot(unit_vectors, optical_axis))
angles_deg = np.degrees(angles)

distortion_center = cam.distortion_center
stretch_matrix = cam.stretch_matrix

cx, cy = distortion_center

x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height))
x_centered = x_coords - cx
y_centered = y_coords - cy

x_rot = x_centered * np.cos(theta) + y_centered * np.sin(theta)
y_rot = -x_centered * np.sin(theta) + y_centered * np.cos(theta)

ellipse_mask = (x_rot**2 / a**2) + (y_rot**2 / b**2) <= 1
masked_angles_deg = np.where(ellipse_mask, angles_deg, np.nan)

max_angle = np.nanmax(masked_angles_deg)
print(f"Maximum angle from optical axis: {max_angle:.2f} degrees")

plt.figure(figsize=(10, 6))

im = plt.imshow(masked_angles_deg, cmap='viridis', vmin=0, vmax=np.nanmax(angles_deg))
contour_levels = np.arange(0, np.nanmax(angles_deg) + 10, 10)
contours = plt.contour(masked_angles_deg, levels=contour_levels, colors='white', linewidths=0.5)

plt.clabel(contours, inline=True, fontsize=8, fmt='%1.0f°')
plt.colorbar(im, label='Angle from optical axis (degrees)')
plt.title('Angle from Optical Axis')
plt.xlabel('X (pixels)')
plt.ylabel('Y (pixels)')
plt.show()