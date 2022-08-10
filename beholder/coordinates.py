import numpy as np


def get_intrinsic_matrix():
    """A helper function to easily swap out the intrinsic matrix. Dependent on
    device calibration

    Returns:
        np.ndarray: _description_
    """
    # TODO add method of automatically determinging intrinsic matrix?
    # fx = 923.40512671  # focal length in pixels, x
    # fy = 922.3138011  # focal length in pixels, y
    fx = 908.4332  # focal length in pixels, x
    fy = 910.0097  # focal length in pixels, y
    ox = 0  # width / 2
    oy = 0  # height / 2
    skew = 0

    intrinsic_matrix = [
        [fx, skew, ox],
        [0,  fy,   oy],
        [0,  0,    1]
    ]
    return intrinsic_matrix


def get_rotation_matrix() -> np.ndarray:
    """A placeholding function that will call some API to retreive the camera
    rotation matrix

    Returns:
        np.ndarray: The 3x3 rotation matrix describing the coordinate rotation
        between the camera reference frame and the world reference frame.
    """
    # TODO replace with API call
    rotation_matrix = np.array([
        [1, 0, 0],
        [0, -1, 0],
        [0, 0, 1]
    ])

    return rotation_matrix


def get_translation_vector() -> np.ndarray:
    """A placeholding function that will call some API to retreive the camera
    translation vector

    Returns:
        np.ndarray: The 3x1 translation vector describing the coordinate
            transform between the camera reference frame and the world
            reference frame.
    """
    # TODO replace with API call
    translation_vec = np.array([0, 0, 0])
    return translation_vec


def get3D_coordinate(point, intrinsic_matrix, rotation_matrix, translation_vec,
                     distance_magnitude) -> list:
    """Converts a 2d pixel coordinate to a 3D world coordinate with predicted
    distances from the model.
    perspective_matrix = intrinsic_matrix * [rotation_matrix | translation_vec]

    Adapted from:
    https://math.stackexchange.com/questions/4382437/back-projecting-a-2d-pixel
    -from-an-image-to-its-corresponding-3d-point

    Args:
        points (tuple): List of points made up of 'u' and 'v' pixel coordinates
            (center of bboxes)
        intrinsic_matrix (np.ndarray): 3x3 matrix describing camera intrinsics
        rotation_matrix (np.ndarray): 3x3 matrix describing camera extrinsics,
            specifically rotation
        translation_vec (np.ndarray): 3x1 x,y,z offset describing camera
            extrinsics, specifically translation
        distance_magnitude (float): the predicted distances corresponding to
            the pixel coordinates

    Returns:
        points3D (tuple): (X,Y,Z) coordinate w.r.t. the camera
    """

    # Homogeneous pixel coordinate
    p = np.array([point[0], point[1], 1]).T

    # Transform pixel in Camera coordinate frame
    pc = np.linalg.inv(intrinsic_matrix) @ p

    # Transform pixel in World coordinate frame
    pw = translation_vec + (rotation_matrix @ pc)

    # Transform camera origin in World coordinate frame
    cam = np.array([0, 0, 0]).T
    cam_world = translation_vec + rotation_matrix @ cam

    # Find a ray from camera to 3d point
    vector = pw - cam_world
    unit_vector = vector / np.linalg.norm(vector)

    # Point scaled along this ray
    p3D = cam_world + distance_magnitude * unit_vector

    return tuple(p3D)


def get_coords_from_tracks(tracks, distances, video_size) -> list:
    """Gets 3D coordinates of targets from list of tracks and predicted
    distances.

    Args:
        tracks (list): list of tracks of detected objects
        distances (list): list of distance predictions.
        video_size (tuple): (x, y) of video size

    Returns:
        list: list of tracks now with 3D coordinate in "point".
    """
    ret_tracks = []

    # done this way to get constant updates from API
    intrinsic_matrix = get_intrinsic_matrix()
    rotation_matrix = get_rotation_matrix()
    translation_vec = get_translation_vector()

    for track, distance in zip(tracks, distances):
        # bbox center + bbox nearest edge from top left corner - half of video
        # size. This finds u, v from the center of the image to the center of
        # the bounding box
        pixel_point = (
            (track['roi'][2] - track['roi'][0])/2
            + track['roi'][0] - video_size[0]/2,
            (track['roi'][3] - track['roi'][1])/2
            + track['roi'][1] - video_size[1]/2
        )  # (u, v)
        track['point'] = get3D_coordinate(pixel_point, intrinsic_matrix,
                                          rotation_matrix, translation_vec,
                                          distance[3])
        ret_tracks.append(track)

    return ret_tracks


if __name__ == '__main__':

    intrinsic_matrix = get_intrinsic_matrix()
    rotation_matrix = get_rotation_matrix()
    translation_vec = get_translation_vector()

    points = (1024, 1024)
    distance_magnitude = 15

    result = get3D_coordinate(points,
                              intrinsic_matrix,
                              rotation_matrix,
                              translation_vec,
                              distance_magnitude)

    print(f'3D vec:    {result}')
    print(f'magnitude: {np.linalg.norm(result)}')
