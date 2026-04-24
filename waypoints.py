#!/Library/Frameworks/Python.framework/Versions/3.11/bin/python3

import numpy as np
import matplotlib.pyplot as plt

def gen_wp(stp, edp, num_points=10, c_type=1):
    if c_type == 1:  # straight line
        x = np.linspace(stp[0], edp[0], num_points)
        y = np.linspace(stp[1], edp[1], num_points)
    elif c_type == 2:  # circle
        center = [(stp[0] + edp[0]) / 2, (stp[1] + edp[1]) / 2]
        radius = np.linalg.norm(np.array(stp) - np.array(edp)) / 2
        angles = np.linspace(0, 2 * np.pi, num_points)
        x = center[0] + radius * np.cos(angles)
        y = center[1] + radius * np.sin(angles)
    elif c_type == 3:  # ellipse
        center = [(stp[0] + edp[0]) / 2, (stp[1] + edp[1]) / 2]
        a = np.linalg.norm(np.array(stp) - np.array(edp)) / 2
        b = a / 2
        angles = np.linspace(0, 2 * np.pi, num_points)
        x = center[0] + a * np.cos(angles)
        y = center[1] + b * np.sin(angles)
    elif c_type == 4:  # infinity
        center = [(stp[0] + edp[0]) / 2, (stp[1] + edp[1]) / 2]
        a = np.linalg.norm(np.array(stp) - np.array(edp)) / 2
        angles = np.linspace(0, 2 * np.pi, num_points)
        x = center[0] + a * np.cos(angles) / (1 + np.sin(angles)**2)
        y = center[1] + a * np.sin(angles) * np.cos(angles) / (1 + np.sin(angles)**2)
    else:
        raise ValueError("Invalid curve type")
    
    return np.array([x, y]).T


if __name__ == "__main__":
    stp = [0, 0]
    edp = [10, 10]
    # straight_line = 1
    # circle = 2
    # ellipse = 3
    # infinity = 4
    waypoints = gen_wp(stp, edp, num_points=100, c_type=1)
    plt.plot(waypoints[:, 0], waypoints[:, 1], marker='o')
    plt.title("Waypoints")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid()
    plt.axis('equal')
    plt.show()

