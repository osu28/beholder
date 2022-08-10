import math


def find_target_ecef(azimuth, distance, camera_ecef):
    """
    calculates the ecef of object with camera at origin
    azimuth refers to degrees between the target and North
    distance is in meters (ecef units)
    """
    z = 0 + camera_ecef[2]
    y = distance * math.cos(90 - azimuth) + camera_ecef[1]
    x = distance * math.sin(90 - azimuth) + camera_ecef[0]
    return (x, y, z)


def camera_to_viewer_movement(camera_ecef, viewer_ecef):
    """
    calculates (x movement, y movement, z movement) needed
    to get from the camera to the viewer
    """
    x_move = viewer_ecef[0] - camera_ecef[0]
    y_move = viewer_ecef[1] - camera_ecef[1]
    z_move = viewer_ecef[2] - camera_ecef[2]
    return (x_move, y_move, z_move)


def center_on_viewer_target(target_xyz, centered_shift):
    """
    calculates the xyz of object with hololens at the origin
    camera_to_viewer represents a transform (1, -2, 0) means 1 right and down 2
    """
    object_xyz = (target_xyz[0] + centered_shift[0],
                  target_xyz[1] + centered_shift[1],
                  target_xyz[2] + centered_shift[2])
    return object_xyz


class BoundingBox(object):
    def __init__(self, xmin=0, ymin=0, xmax=0, ymax=0):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.centroid = ((xmin + xmax)/2, (ymin + ymax)/2)


class Position(object):
    def __init__(self, coord=(0, 0, 0), ecef=(0, 0, 0), xyz=(0, 0, 0)):
        # coord follows geodetic (degrees N, degrees W, height)
        self.coord = coord
        # ecef
        self.ecef = ecef
        # used to render follows cartesian 3D
        # (left/right x, up/down y, towards/away z)
        self.xyz = xyz
