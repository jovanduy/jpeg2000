###################################
# JPEG2000 Image Compression script
###################################

from PIL import Image
from pickle import dump
import numpy as np
import cv2
import pywt

class Tile(object):
    """tile class for storing original tile image, Y, Cr and Cb images"""
    def __init__(self, tile_image):
        self.tile_image = tile_image
        self.y_tile, self.Cb_tile, self.Cr_tile = None, None, None

class JPEG2000(object):
    """compression algorithm, jpeg2000"""
    def __init__(self):   
        self.file_path = "data/image.jpg"
        self.debug = True
        self.tiles = []
        self.component_transformation_matrix = np.array([[0.2999, 0.587, 0.114],
            [-0.16875, -0.33126, 0.5],[0.5, -0.41869, -0.08131]])
        self.inverse_component_transformation_matrix = ([[1.0, 0, 1.402], [1.0, -0.34413, -0.71414], [1.0, 1.772, 0]])
        self.tile_size = 300

    def init_image(self, path):
        img = cv2.imread(path)
        # imgYCC = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
        # print type(imgYCC)
        # print imgYCC.shape
        # print imgYCC[:, :, 0].shape
        # cv2.imshow("imgYCC", imgYCC[:, :, 0])
        # cv2.waitKey(0) 
        # raise "debug"       
        return img

    def image_tiling(self, img):
        # tile image
        tile_size = self.tile_size
        (h, w, _) = img.shape
        counter = 0
        
        left_over = w%tile_size
        w += (tile_size - left_over)
        left_over = h%tile_size
        h += (tile_size - left_over)

        for i in range(0, w, tile_size):    # for every pixel:
            for j in range(0, h, tile_size):
                tile = Tile(img[j:j + tile_size, i:i + tile_size])
                counter += 1
                self.tiles.append(tile)

                # if self.debug:
                #     cv2.imshow("test_image" + str(counter), tile.tile_image)
                #     cv2.waitKey(0)

    def dc_level_shift(self, img):
        # dc level shifting

        pass
    
    def component_transformation(self):
        # component transformation:
        # split image into Y, Cb, Cr

        for tile in self.tiles:
            (h, w, _) = tile.tile_image.shape
            rgb_tile = cv2.cvtColor(tile.tile_image, cv2.COLOR_BGR2RGB)
            Image_tile = Image.fromarray(rgb_tile, 'RGB')
            # tile.y_tile, tile.Cb_tile, tile.Cr_tile = np.empty_like(tile.tile_image), np.empty_like(tile.tile_image), np.empty_like(tile.tile_image)

            # tile.y_tile, tile.Cb_tile, tile.Cr_tile = np.zeros_like(tile.tile_image), np.zeros_like(tile.tile_image), np.zeros(tile.tile_image)
            tile.y_tile, tile.Cb_tile, tile.Cr_tile = np.zeros((h, w)), np.zeros((h, w)), np.zeros((h, w))


            for i in range(0, w):    # for every pixel:
                for j in range(0, h):
                    r, g, b = Image_tile.getpixel((i, j))
                    rgb_array = np.array([r, g, b])
                    yCbCr_array = np.matmul(self.component_transformation_matrix, rgb_array)
                    # y = .299 * r + .587 * g + .114 * b 
                    # Cb = 0 
                    # Cr = 0
                    tile.y_tile[j][i], tile.Cb_tile[j][i], tile.Cr_tile[j][i] = int(yCbCr_array[0]), int(yCbCr_array[1]), int(yCbCr_array[2])
                    # tile.y_tile[j][i], tile.Cb_tile[j][i], tile.Cr_tile[j][i] = int(y), int(Cb), int(Cr)
            
            # if self.debug:
            #     print tile.y_tile.shape
            #     cv2.imshow("y_tile", tile.y_tile)
            #     cv2.imshow("Cb_tile", tile.Cb_tile)
            #     cv2.imshow("Cr_tile", tile.Cr_tile)
            #     # print tile.y_tile[0]
            #     cv2.waitKey(0)

    def inverse_component_transformation(self):
        # component transformation:
        # split image into Y, Cb, Cr

        for tile in self.tiles:
            (h, w, _) = tile.tile_image.shape
            tile.recovered_tile = np.empty_like(tile.tile_image)

            for i in range(0, w):    # for every pixel:
                for j in range(0, h):
                    y, Cb, Cr = tile.recovered_y_tile[j][i], tile.recovered_Cb_tile[j][i], tile.recovered_Cr_tile[j][i]
                    
                    yCbCr_array = np.array([y, Cb, Cr])

                    rgb_array = np.matmul(self.inverse_component_transformation_matrix, yCbCr_array)
                    tile.recovered_tile[j][i] = rgb_array
            # break
            if self.debug:
                rgb_tile = cv2.cvtColor(tile.recovered_tile, cv2.COLOR_RGB2BGR)
                cv2.imshow("tile.recovered_tile", rgb_tile)
                cv2.waitKey(0)

    def dwt(self):
        # do the mathmagic dwt
        for tile in self.tiles:
            # print "before dwt tile.y_tile.shape: ", tile.y_tile.shape
            tile.y_coeffs = pywt.dwt2(tile.y_tile, 'haar') #cA, (cH, cV, cD) 
            cA, (cH, cV, cD) = tile.y_coeffs
            # print type(cA)
            # print cA.shape
            tile.Cb_coeffs = pywt.dwt2(tile.Cb_tile, 'haar')
            tile.Cr_coeffs = pywt.dwt2(tile.Cr_tile, 'haar')

        # tile = self.tiles[0]
        # print type(tile.y_coeffs[0])
        # print tile.y_coeffs[0].shape
        # print tile.y_coeffs[0]
        # cv2.imshow("y_tile", tile.y_coeffs[0])
        # cv2.waitKey(0)
    def idwt(self):
        for tile in self.tiles:
            tile.recovered_y_tile = pywt.idwt2(tile.y_coeffs, 'haar')  
            tile.recovered_Cb_tile = pywt.idwt2(tile.Cb_coeffs, 'haar')  
            tile.recovered_Cr_tile = pywt.idwt2(tile.Cr_coeffs, 'haar')  
            # break
        # print tile.recovered_y_tile.shape
        # print tile.recovered_Cb_tile.shape
        # print tile.recovered_Cr_tile.shape
        # tile = self.tiles[0] 
        # print tile.y_tile[0]
        # print tile.recovered_y_tile[0]


    def quantization(self, img):
        # quantization
        pass

    def entropy_coding(self, img):
        # encode image
        pass

    def bit_stream_formation(self, img):
        # idk if we need this or what it is
        pass

    def forward(self):
        img = self.init_image(self.file_path)
        self.image_tiling(img)
        self.component_transformation()
        self.dwt()

    def backward(self):
        self.idwt()
        self.inverse_component_transformation()

    def run(self):
        self.forward()
        self.backward()


if __name__ == '__main__':
    JPEG2000().run()