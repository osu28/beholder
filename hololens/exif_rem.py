# helper file for investigating photo metadata

from PIL.ExifTags import GPSTAGS
from PIL import Image
from PIL.ExifTags import TAGS


def get_exif(filename):
    image = Image.open(filename)
    image.verify()
    return image._getexif()


def get_labeled_exif(exif):
    labeled = {}
    for (key, val) in exif.items():
        labeled[TAGS.get(key)] = val

    return labeled


def get_geotagging(exif):
    if not exif:
        raise ValueError("No EXIF metadata found")

    geotagging = {}
    for (idx, tag) in TAGS.items():
        if tag == 'GPSInfo':
            if idx not in exif:
                raise ValueError("No EXIF geotagging found")

            for (key, val) in GPSTAGS.items():
                if key in exif[idx]:
                    geotagging[val] = exif[idx][key]

    return geotagging


exif = get_exif('image2.jpg')
geotags = get_geotagging(exif)
for item in geotags.items():
    print(item)
