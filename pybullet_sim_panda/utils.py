import numpy as np
import pybullet as p
from spatial_math_mini import SE3

"""
Utility Functions.
- view_pose(pos, ori, color=None)
"""
def view_pose(pos, ori, color=None):
    """show the debugging purpose 3-axis frame using pos/ori information

    :param pos [array-like(3)]: The position of the frame
    :param ori [array-like(4)]: The orientation of the frame
    :param color: specific color, defaults to None
    :return [list]: object ids
    """
    T = SE3.qtn_trans(ori, pos).T
    length = 0.1
    xaxis = np.array([length, 0, 0, 1])
    yaxis = np.array([0, length, 0, 1])
    zaxis = np.array([0, 0, length, 1])
    T_axis = np.array([xaxis, yaxis, zaxis]).T
    axes = T @ T_axis
    orig = T[:3,-1]
    xaxis = axes[:-1,0]
    yaxis = axes[:-1,1]
    zaxis = axes[:-1,2]
    if color == "r":
        x_color = y_color = z_color = [1,0,0]
    elif color == "g":
        x_color = y_color = z_color = [0,1,0]
    elif color == "b":
        x_color = y_color = z_color = [0,0,1]
    elif color == "k":
        x_color = y_color = z_color = [0,0,0]
    else:
        x_color = [1,0,0]
        y_color = [0,1,0]
        z_color = [0,0,1]

    x = p.addUserDebugLine(orig, xaxis, lineColorRGB=x_color, lineWidth=5)
    y = p.addUserDebugLine(orig, yaxis, lineColorRGB=y_color, lineWidth=5)
    z = p.addUserDebugLine(orig, zaxis, lineColorRGB=z_color, lineWidth=5)
    pose_id = [x, y, z]
    return pose_id

def view_point(pos):
    """show the debugging purpose 3-axis frame using only position information

    :param pos [array-like(3)]: the position of the frame
    """
    view_pose(pos, [1,0,0,0])

def clear(rng=None):
    """[summary]

    :param rng [array-like]: a list of object numbers to remove. if None, delete 0~100, defaults to None
    """
    if rng is None:
        rng = range(100)
    for i in rng:
        p.removeUserDebugItem(i)