###################################
# JPEG2000 Image Compression script
###################################

from PIL import Image
from pickle import dump
import numpy as np
import cv2

class JPEG2000(object):
    """compression algorithm, jpeg2000"""
    def __init__(self):   
        self.file_path = "data/image.jpg"
        self.debug = True
        self.tiles = []

    def init_image(self, path):
        img = cv2.imread(path)
        return img

    def image_tiling(self, img, tile_size = 8):
        # tile image
        (h, w, _) = img.shape
        print img.shape
        counter = 0
        
        left_over = w%tile_size
        w += (tile_size - left_over)
        left_over = h%tile_size
        h += (tile_size - left_over)

        for i in range(0, w, tile_size):    # for every pixel:
            for j in range(0, h, tile_size):
                print j, i
                tile = img[j:j + tile_size, i:i + tile_size]
                print type(tile)
                print tile.shape
                counter += 1
                self.tiles.append(tile)

                if self.debug:
                    cv2.imshow("test_image" + str(counter), tile)
                    cv2.waitKey(0)

    def dc_level_shift(self, img):
        # dc level shifting

        pass
    
    def component_transformation(self, img):
        # component transformation:
        # split image into Y, Cb, Cr
        for tile in self.tiles:
            (h, w, _) = tile.shape
            tile = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)
            tile = Image.fromarray(tile, 'RGB')
            for i in range(0, w):    # for every pixel:
                for j in range(0, h):
                    img.getpixel(i, j)


        # img.save('my.jpg')
        # img.show()        
        # componentMatrix = [[],[],[]]
        pass

    def dwt(self, img):
        # do the mathmagic dwt
        pass

    def quantization(self, img):
        # quantization
        pass

    def entropy_coding(self, img):
        # encode image
        pass

    def bit_stream_formation(self, img):
        # idk if we need this or what it is
        pass

    def forward():
        pass
    def backward():
        pass

    def run(self):
        img = self.init_image(self.file_path)
        # self.image_tiling(img, tile_size = 300)

if __name__ == '__main__':
    JPEG2000().run()