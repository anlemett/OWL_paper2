import os
import numpy as np
import matplotlib.pyplot as plt
import csv

FIG_DIR = os.path.join(".", "Figures")


# Read the coordinates from the CSV file
x = []
y = []

with open('curve_coordinates.csv', mode='r') as file:
    reader = csv.reader(file)
    next(reader)  # Skip the header
    for row in reader:
        x.append(float(row[0]))
        y.append(float(row[1]))


# Plot accuracies
fig0, ax0 = plt.subplots()
ax0.plot(x, y, marker='o')


# Get the current axis and its limits
#ax = plt.gca()
xlim = ax0.get_xlim()
ylim = ax0.get_ylim()

# Determine the data ranges
x_range = xlim[1] - xlim[0]
y_range = ylim[1] - ylim[0]

# Get the figure size in inches and DPI (dots per inch)
fig_width, fig_height = plt.gcf().get_size_inches()
fig_dpi = plt.gcf().get_dpi()

# Calculate the pixel dimensions of the plot
pixel_width = fig_width * fig_dpi
pixel_height = fig_height * fig_dpi

# Calculate scale factors for each axis
x_scale = pixel_width / x_range
y_scale = pixel_height / y_range

# Define the two points that determine the line
A = np.array([x[0] * x_scale, y[0] * y_scale])
B = np.array([x[-1] * x_scale, y[-1] * y_scale])

elbow_point_num = 9
elbow_point_idx = 8
 
# Define the point from which the perpendicular is drawn
P = np.array([elbow_point_num * x_scale, y[elbow_point_idx] * y_scale])
 
# Vector AB
AB = B - A
 
# Vector AP
AP = P - A

# Project vector AP onto AB to find the perpendicular foot
t = np.dot(AP, AB) / np.dot(AB, AB)
F = A + t * AB  # F is the perpendicular foot

F = np.array([F[0] / x_scale, F[1] / y_scale])
A = np.array([A[0] / x_scale, A[1] / y_scale])
B = np.array([B[0] / x_scale, B[1] / y_scale])
P = np.array([P[0] / x_scale, P[1] / y_scale])

light_red = (1.0, 0.4, 0.4)  # RGB values for a lighter red
light_green = (0.4, 0.8, 0.4)  # RGB values for a lighter green
ax0.plot([A[0], B[0]], [A[1], B[1]], '--', color=light_green, label='Line AB')  # Line AB
ax0.plot([P[0], F[0]], [P[1], F[1]], '--', color=light_red, label='Perpendicular')  # Perpendicular line
ax0.plot(A[0], A[1], 'o', color=light_green, label='Point A')
ax0.plot(B[0], B[1], 'o', color=light_green, label='Point B')
ax0.plot(P[0], P[1], 'o', color=light_red, label='Point P')
#ax0.plot(F[0], F[1], 'ro', label='Foot of Perpendicular')
 
ax0.set_xlabel('Number of Features', fontsize=14)
ax0.set_ylabel('Accuracy', fontsize=14)
ax0.tick_params(axis='both', which='major', labelsize=12)
ax0.tick_params(axis='both', which='minor', labelsize=10)
plt.grid(True)
filename = "f1_scoring_eeg_3classes_HGBC_acc2.png"
full_filename = os.path.join(FIG_DIR, filename)

plt.savefig(full_filename, dpi=600)        
plt.show()