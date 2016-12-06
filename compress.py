###################################
# JPEG2000 Image Compression script
###################################

from PIL import Image
from pickle import dump

file_path = "data/image.jpg"

def init_image(path):
    return Image.open(path)

def image_tiling(img):
    # tile image

def dc_level_shift(img):
    # dc level shifting

def component_transformation(img):
    # component transformation:
    # split image into Y, Cb, Cr

def dwt(img):
    # do the mathmagic dwt

def quantization(img):
    # quantization

def entropy_coding(img):
    # encode image

def bit_stream_formation(img):
    # idk if we need this or what it is

if __name__ == '__main__':
    img = init_image(file_path)
