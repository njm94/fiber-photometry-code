import numpy as np
import re
import os

def quantify_time(x, y, loc):
    if loc[0] == 'U':
        y_corner = y < (np.nanmin(y) + 1/2 * (np.nanmax(y) - np.nanmin(y)))
    elif loc[0] == 'L':
        y_corner = y > (np.nanmin(y) + 1/2 * (np.nanmax(y) - np.nanmin(y)))
    else:
        print("Invalid location input for Y")

    if loc[1] == 'L':
        x_corner = x < (np.nanmin(x) + 1/2 * (np.nanmax(x) - np.nanmin(x)))
    elif loc[1] == 'R':
        x_corner = x > (np.nanmin(x) + 1/2 * (np.nanmax(x) - np.nanmin(x)))
    else:
        print("Invalid location input for X")

    x_center = (x > (np.nanmin(x) + 1/4 * (np.nanmax(x) - np.nanmin(x)))) & (x < (np.nanmin(x) + 3/4 * (np.nanmax(x) - np.nanmin(x))))
    y_center = (y > (np.nanmin(y) + 3/4 * (np.nanmax(y) - np.nanmin(y)))) & (x < (np.nanmin(y) + 3/4 * (np.nanmax(y) - np.nanmin(y))))

    corner_time = 100*sum(x_corner & y_corner)/len(x)
    center_time = 100*sum(x_center & y_center)/len(x)

    return corner_time, center_time





# wd = r'Y:\nick\behavior\open_field\avoidance\bobcat\urine\outputs'
wd = r'Y:\nick\behavior\open_field\avoidance\bobcat\water_control\outputs'

for fname in os.listdir(wd):
    if fname.endswith('_tracks.npy'):
        tracks = np.load(wd + os.path.sep + fname)
        loc = re.findall(r"-.{2}-", fname)[0][1:-1]
        corner_time, center_time = quantify_time(tracks[0, :], tracks[1, :], loc)
        print(fname, corner_time, center_time)
