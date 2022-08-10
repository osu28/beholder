import numpy as np
from PIL import Image


# resize transform for our data and boxes
class Resize(object):
    def __init__(self, size):
        self.w, self.h = size[0], size[1]

    def __call__(self, image, boxes):

        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        w, h = image.size
        nw, nh = self.w, self.h

        # resize image
        scale = min(float(nw) / w, float(nh) / h)
        size = (int(w * scale), int(h * scale))
        image = image.resize(size)

        # convert to numpy
        image = np.asarray(image)

        # pad image
        padded_img = np.ones((nh, nw, 3), dtype=np.uint8) * 114
        padded_img[:size[1], :size[0]] = image
        image = padded_img

        # resize boxes
        boxes = resize_boxes(boxes, w, h, size[0], size[1])

        return image, boxes


# function to resize boxes
def resize_boxes(bboxes, x_size, y_size, new_x_size, new_y_size):

    # find the ratio of the sizes
    x_scale, y_scale = new_x_size/x_size, new_y_size/y_size

    # multiply by bbox coordinates
    bboxes = bboxes.astype(float)
    bboxes[:, [0]] *= x_scale
    bboxes[:, [2]] *= x_scale
    bboxes[:, [1]] *= y_scale
    bboxes[:, [3]] *= y_scale

    return bboxes
