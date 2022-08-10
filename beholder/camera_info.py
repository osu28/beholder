#!/usr/bin/env python
# this script is necessary for getting camera intrinsics, but needs to be
# rewritten to use images from the RTMP stream

import cv2
import numpy as np
import time
from twitchrealtimehandler import TwitchImageGrabber


class TwitchStream():
    """integrates the streaming object into MOT since the TwitchImageGrabber
    is not a video reader object (which MOT was written for)
    """

    def __init__(self, url, quality='720p'):
        self.url = url
        self.fps = 30
        self.video = TwitchImageGrabber(
            twitch_url=url,
            quality=quality,
            blocking=True,
            rate=self.fps  # frame per rate (fps)
        )
        self.src_size = (self.video.width, self.video.height)
        self.size = self.src_size
        self._is_stream = True
        self.frame_start = 0

    def read(self):
        return self.video.grab()


if __name__ == '__main__':
    # Defining the dimensions of checkerboard
    CHECKERBOARD = (6, 9)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Creating vector to store vectors of 3D points for each checkerboard image
    objpoints = []
    # Creating vector to store vectors of 2D points for each checkerboard image
    imgpoints = []

    stream = TwitchStream(url="https://www.twitch.tv/intern_project")

    # Defining the world coordinates for 3D points
    objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0],
                              0:CHECKERBOARD[1]].T.reshape(-1, 2)
    prev_img_shape = None

    # Extracting path of individual image stored in a given directory
    for i in range(20):
        img = stream.read()
        print(f'{i}')
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # Find the chess board corners
        # If desired number of corners are found in the image then ret = true
        ret, corners = cv2.findChessboardCorners(
            gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH +
            cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

        """
        If desired number of corner are detected,
        we refine the pixel coordinates and display
        them on the images of checker board
        """
        if ret is True:
            print('valid')
            objpoints.append(objp)
            # refining pixel coordinates for given 2d points.
            corners2 = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1), criteria)

            imgpoints.append(corners2)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        else:
            print('invalid')

        # cv2.imshow('img',img)
        t0 = time.time()
        while time.time() - t0 < 1:
            img = stream.read()

    h, w = img.shape[:2]

    """
    Performing camera calibration by
    passing the value of known 3D points (objpoints)
    and corresponding pixel coordinates of the
    detected corners (imgpoints)
    """
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None)

    print("Camera matrix : \n")
    print(mtx)
    print("dist : \n")
    print(dist)
    print("rvecs : \n")
    print(rvecs)
    print("tvecs : \n")
    print(tvecs)
