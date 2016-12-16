"""""""""""""""""""""""""""""""""""
JPEG2000 Image Compression script
to run: python compress.py
"""""""""""""""""""""""""""""""""""

from PIL import Image
from pickle import dump
import numpy as np
import cv2
import pywt
import math

class Tile(object):
    """ Tile class for storing data about a tile of an image """
    def __init__(self, tile_image):
        """
        tile_image: original tile image (np.array)
        y_tile: original tile's y-colorspace (np.array)
        Cb_tile: original tile's Cb-colorspace (np.array)
        Cr_tile: original tile's Cr-colorspace (np.array)
        """
        self.tile_image = tile_image
        self.y_tile, self.Cb_tile, self.Cr_tile = None, None, None

class JPEG2000(object):
    """compression algorithm, jpeg2000"""
    def __init__(self, file_path = "data/test.jpg", quant = True, lossy = True, tile_size = 300):
        """ 
        Initialize JPEG2000 algorithm 
        
        Initial parameters:
        file_path: path to image file to be compressed (string)
        quant: include quantization step (boolean)
        lossy: perform lossy compression (boolean)
        tile_size: size of tile (int)
        """
        self.file_path = file_path
        self.debug = True
        self.lossy = lossy

        # list of Tile objects of image
        self.tiles = []
        
        # tile size
        self.tile_size = tile_size

        # lossy compression component transform matrices
        self.component_transformation_matrix = np.array([[0.2999, 0.587, 0.114],
            [-0.16875, -0.33126, 0.5],[0.5, -0.41869, -0.08131]])
        self.i_component_transformation_matrix = ([[1.0, 0, 1.402], [1.0, -0.34413, -0.71414], [1.0, 1.772, 0]])  

       # haar filter coefficients
        self.h0 = [math.sqrt(0.5), math.sqrt(0.5)]
        self.h1 = [math.sqrt(0.5),  - math.sqrt(0.5)]

        # debauchies
        # self.h0 = [-0.010597401784997278, 0.032883011666982945, 0.030841381835986965, -0.18703481171888114, -0.02798376941698385, 0.6308807679295904, 0.7148465705525415, 0.23037781330885523];
        # self.h1 = [-0.2303778133, 0.7148465706, -0.6308807679, -0.0279837694, 0.1870348117, 0.0308413818, -0.0328830117, -0.0105974018]

        # quantization
        self.quant = quant
        self.step = 30


    def init_image(self, path):
        """ return the image at path """
        img = cv2.imread(path)
        return img

    def image_tiling(self, img):
        """ 
        tile img into square tiles based on self.tile_size
        tiles from bottom and right edges will be smaller if
        image w and h are not divisible by self.tile_size
        """
        tile_size = self.tile_size
        (h, w, _) = img.shape # size of original image
        counter = 0
        
        # change w and h to be divisible by tile_size
        left_over = w % tile_size
        w += (tile_size - left_over)
        left_over = h % tile_size
        h += (tile_size - left_over)

        # create the tiles by looping through w and h to stop on 
        # every pixel that is the top left corner of a tile
        for i in range(0, w, tile_size): # loop through the width (columns) of img, skipping tile_size pixels every time
            for j in range(0, h, tile_size): # loop through the height (rows) of img, skipping tile_size pixels every time
                # add the tile starting at pixel of row j and column i
                tile = Tile(img[j:j + tile_size, i:i + tile_size])

                self.tiles.append(tile)

                if self.debug:
                    cv2.imshow("tile" + str(counter), tile.tile_image)
                    cv2.waitKey(0)
                    cv2.imwrite("tile " + str(counter) + ".jpg", tile.tile_image)

                    counter += 1

    def dc_level_shift(self, img):
        # dc level shifting
        pass
    
    def component_transformation(self):
        """
        Transform every tile in self.tiles from RGB colorspace
        to either YCbCr colorspace (lossy) or YUV colorspace (lossless)
        and save the data for each color component into the tile object
        """
        # loop thorugh tiles
        for tile in self.tiles:
            (h, w, _) = tile.tile_image.shape # size of tile

            # transform tile to RGB colorspace (library we use to view images uses BGR)
            rgb_tile = cv2.cvtColor(tile.tile_image, cv2.COLOR_BGR2RGB)
            Image_tile = Image.fromarray(rgb_tile, 'RGB')
            
            # create placeholder matrices for the different colorspace components
            # that are same w and h as original tile
            #tile.y_tile, tile.Cb_tile, tile.Cr_tile = np.empty_like(tile.tile_image), np.empty_like(tile.tile_image), np.empty_like(tile.tile_image)
            tile.y_tile, tile.Cb_tile, tile.Cr_tile = np.zeros((h, w)), np.zeros((h, w)), np.zeros((h, w))
            # tile.y_tile, tile.Cb_tile, tile.Cr_tile = np.zeros_like(tile.tile_image), np.zeros_like(tile.tile_image), np.zeros_like(tile.tile_image)

            # loop through every pixel and extract the corresponding
            # transformed colorspace values and save in tile object
            for i in range(0, w):
                for j in range(0, h):
                    r, g, b = Image_tile.getpixel((i, j))
                    rgb_array = np.array([r, g, b])
                    if self.lossy:
                        # use irreversible component transformation matrix to transform to YCbCr
                        yCbCr_array = np.matmul(self.component_transformation_matrix, rgb_array)
                    else:
                        # use reversible component transform to get YUV components
                        yCbCr_array = np.matmul(self.component_transformation_matrix, rgb_array)

                    # y = .299 * r + .587 * g + .114 * b 
                    # Cb = 0 
                    # Cr = 0
                    tile.y_tile[j][i], tile.Cb_tile[j][i], tile.Cr_tile[j][i] = int(yCbCr_array[0]), int(yCbCr_array[1]), int(yCbCr_array[2])
                    # tile.y_tile[j][i], tile.Cb_tile[j][i], tile.Cr_tile[j][i] = int(y), int(Cb), int(Cr)
            
        if self.debug:
            tile = self.tiles[0]
            print tile.y_tile.shape
            Image.fromarray(tile.y_tile).show()


        #     # Image.fromarray(tile.y_tile).convert('RGB').save("my.jpg")


        #     # cv2.imshow("y_tile", tile.y_tile)
        #     # cv2.imshow("Cb_tile", tile.Cb_tile)
        #     # cv2.imshow("Cr_tile", tile.Cr_tile)
        #     # print tile.y_tile[0]
        #     cv2.waitKey(0)
        

    def i_component_transformation(self):
        """
        Inverse component transformation:
        transform all tile back to RGB colorspace
        """
        # loop through tiles, converting each back to RGB colorspace
        for tile in self.tiles:
            (h, w, _) = tile.tile_image.shape # size of tile
            # (h, w) = tile.recovered_y_tile.shape

            # initialize recovered tile matrix to same size as original 3 dimensional tile
            tile.recovered_tile = np.empty_like(tile.tile_image)

            # loop through every pixel of the tile recovered from iDWT and use
            # the YCbCr values (if lossy) or YUV values (is lossless)
            # to transfom back to single RGB tile
            for i in range(0, w):
                for j in range(0, h):
                    y, Cb, Cr = tile.recovered_y_tile[j][i], tile.recovered_Cb_tile[j][i], tile.recovered_Cr_tile[j][i]
                    yCbCr_array = np.array([y, Cb, Cr])
                    
                    if self.lossy:
                        # use irreversible component transform matrix to get back RGB values
                        rgb_array = np.matmul(self.i_component_transformation_matrix, yCbCr_array)
                    else:
                        # use reversible component transform to get back RGB values
                        rgb_array = np.matmul(self.i_component_transformation_matrix, yCbCr_array)
                    # save all three color dimensions to the given pixel
                    tile.recovered_tile[j][i] = rgb_array
            # break
                
            # if self.debug:
            #     rgb_tile = cv2.cvtColor(tile.recovered_tile, cv2.COLOR_RGB2BGR)
            #     print "rgb_tile.shape: ", rgb_tile.shape
            #     cv2.imshow("tile.recovered_tile", rgb_tile)
            #     cv2.waitKey(0)



    def DWT(self, level = 1):
        """
        Use the discrete wavelet transform to get coefficients for the three different
        components of each tile, saving coefficients in the tile image
        level: number of times to run DWT (using lowpass approx for subsequent calls)
        """
        # loop through all of the tiles
        for tile in self.tiles:
            tile.y_coeffs  = self.DWT_helper(tile.y_tile, level)
            tile.Cb_coeffs = self.DWT_helper(tile.Cb_tile, level)
            tile.Cr_coeffs = self.DWT_helper(tile.Cr_tile, level)

        if self.debug:
            tile = self.tiles[0]
            # print "tile.y_coeffs[3].shape: ", tile.y_coeffs[3].shape
            cv2.imshow("tile.y_coeffs[0]", tile.y_coeffs[0])
            cv2.imshow("tile.y_coeffs[1]", tile.y_coeffs[1])
            cv2.imshow("tile.y_coeffs[2]", tile.y_coeffs[2])
            cv2.imshow("tile.y_coeffs[3]", tile.y_coeffs[3])
            Image.fromarray(tile.y_coeffs[0]).show()
            Image.fromarray(tile.y_coeffs[1]).show()
            Image.fromarray(tile.y_coeffs[2]).show()
            Image.fromarray(tile.y_coeffs[3]).show()


            # from PIL import Image
            # img = Image.fromarray(tile.y_coeffs[3], 'RGB')
            # img.save('my.png')
            # img.show()
            cv2.waitKey(0)

    def iDWT(self, level = 1):
        """
        Use the inverse discrete wavelet transform to recover the pixel
        values for each color component of every tile, saving values in
        the tile image
        level: number of tiles to run iDWT
        """
        # loop through all of the tiles
        for tile in self.tiles:
            if self.quant:
                # if tile was quantized, need to use the recovered, un-quantized coefficients
                tile.recovered_y_tile  = self.iDWT_helper(tile.recovered_y_coeffs, level)
                tile.recovered_Cb_tile = self.iDWT_helper(tile.recovered_Cb_coeffs, level)
                tile.recovered_Cr_tile = self.iDWT_helper(tile.recovered_Cr_coeffs, level)
            else:
                # if tile wasn't quantized, need to use the coeffs from DWT
                tile.recovered_y_tile  = self.iDWT_helper(tile.y_coeffs, level)
                tile.recovered_Cb_tile = self.iDWT_helper(tile.Cb_coeffs, level)
                tile.recovered_Cr_tile = self.iDWT_helper(tile.Cr_coeffs, level)
            
            # break

        # if self.debug:
        #     tile = self.tiles[0]        
        #     print "tile.recovered_y_tile.shape: ", tile.recovered_y_tile.shape
        #     print "tile.recovered_Cb_tile.shape: ", tile.recovered_Cb_tile.shape
        #     print "tile.recovered_Cr_tile.shape: ", tile.recovered_Cr_tile.shape
        #     # cv2.imshow("tile.recovered_y_tile", tile.recovered_y_tile)
        #     Image.fromarray(tile.recovered_y_tile).show()

        

    def DWT_helper(self, img, level):
        """
        Impletement the DWT using convolution on img
        """
        (h, w) = img.shape # size of image
        print "img.shape ", img.shape
        # placeholder arrays for coefficients resulting from first run
        # of high and low pass filtering, along with downsampling
        highpass = []
        lowpass = []
        highpass_down = []
        lowpass_down = []
        
        # convolve the rows
        for row in range(h):
            highpass.append(np.convolve(img[row,:], self.h1[::-1]))
            lowpass.append(np.convolve(img[row,:], self.h0[::-1]))

        # turn highpass and lowpass into np.arrays
        # to allow for indexing by column
        highpass = np.asarray(highpass)
        lowpass = np.asarray(lowpass)

        print "highpass.shape: ", highpass.shape
        print "lowpass.shape: ", lowpass.shape
        
        # downsample the rows
        for row in range(h):
            highpass_down.append(highpass[row][::2])
            lowpass_down.append(lowpass[row][::2])

        # turn into np.arrays
        highpass_down = np.asarray(highpass_down)
        lowpass_down = np.asarray(lowpass_down)

        print "highpass_down.shape: ", highpass_down.shape
        print "lowpass_down.shape: ", lowpass_down.shape

        # size of downsampled, filtered once tile
        (h, w) = highpass_down.shape

        # initialize arrays for final coefficients after 2D filtering
        hh = []
        hl = []
        ll = []
        lh = []

        # initialize arrays for coefficients after final downsampling
        hh_down = []
        hl_down = []
        lh_down = []
        ll_down = []

        # convolute the columns, appending each column as a sub array to the given array
        for col in range(w):
            hh.append(np.convolve(highpass_down[:,col], self.h1[::-1]))
            hl.append(np.convolve(highpass_down[:,col], self.h0[::-1]))
            lh.append(np.convolve(lowpass_down[:,col], self.h1[::-1]))
            ll.append(np.convolve(lowpass_down[:,col], self.h0[::-1]))

        # turn the arrays to np.arrays and transpose them
        # (since columns were appended as rows in above step)
        hh = np.transpose(np.asarray(hh))
        hl = np.transpose(np.asarray(hl))
        lh = np.transpose(np.asarray(lh))
        ll = np.transpose(np.asarray(ll))


        print "hh.shape: ", hh.shape
        print "hl.shape: ", hl.shape
        print "lh.shape: ", lh.shape
        print "ll.shape: ", ll.shape

        # downsample the columns
        for col in range(w):
            hh_down.append(hh[:, col][::2])
            hl_down.append(hl[:, col][::2])
            lh_down.append(lh[:, col][::2])
            ll_down.append(ll[:, col][::2])

        # turn the arrays to np.arrays and transpose them
        # (since columns were appended as rows in above step)
        hh_down = np.transpose(np.asarray(hh_down))
        hl_down = np.transpose(np.asarray(hl_down))
        lh_down = np.transpose(np.asarray(lh_down))
        ll_down = np.transpose(np.asarray(ll_down))

        print "hh_down.shape: ", hh_down.shape
        print "hl_down.shape: ", hl_down.shape
        print "lh_down.shape: ", lh_down.shape
        print "ll_down.shape: ", ll_down.shape

        # run the DWT again on lowpass approximation for deeper filtering
        if (level > 1):
            hh_down, hl_down, lh_down, ll_down = DWT_helper(ll_down, level-1)

        return (hh_down, hl_down, lh_down, ll_down)

    def iDWT_helper(self, img, level):
        """
        Implement the inverse DWT on the tiles
        of img, using convolution
        """
        print "------------------------------------"

        # the 4 arrays holding the coefficients for the iDWT
        hh = img[0]
        hl = img[1]
        lh = img[2]
        ll = img[3]

        # initialize arrays for upsampling
        i_hh_up = []
        i_hl_up = []
        i_lh_up = []
        i_ll_up = []

        (h, w) = i_hh.shape
        print "initial state"
        print "initial sample size: ", i_hh.shape

        # up sampling on the rows (append a row of zeros after every row)
        for row in range(h):
            i_hh_up.append(hh[row])
            i_hh_up.append(np.zeros(w))

            i_hl_up.append(hl[row])
            i_hl_up.append(np.zeros(w))

            i_lh_up.append(lh[row])
            i_lh_up.append(np.zeros(w))

            i_ll_up.append(ll[row])
            i_ll_up.append(np.zeros(w))

        print "finish up sampling, ", np.asarray(i_hh_up).shape
        
        # transform upsampled arrays to np.arrays
        i_hh_up = np.asarray(i_hh_up)
        i_hl_up = np.asarray(i_hl_up)
        i_lh_up = np.asarray(i_lh_up)
        i_ll_up = np.asarray(i_ll_up)

        (h, w) = i_hh_up.shape

        # initialize arrays to hold coefficients that will eventually
        # be put through the high and low pass filters, respectively
        highpass = []
        lowpass = []

        # convolve columns and sum diagonal and vertical, horizontal and LP approx
        # and then append to new matrices
        for col in range(w):
            # diagonal (HP) and vertical (LP)
            convolution_result = np.convolve(i_hh_up[:, col], self.h1) + np.convolve(i_hl_up[:, col], self.h0)
            highpass.append(convolution_result)
            
            # print " first convolution_result ", convolution_result.shape
            # print len(highpass)
            # print len(highpass[-1])

            # horizontal (HP) and LP approx (LP)
            convolution_result = np.convolve(i_lh_up[:, col], self.h1) + np.convolve(i_ll_up[:, col], self.h0)
            lowpass.append(convolution_result)

        # change to np.array and transpose the matrices
        highpass = np.transpose(np.asarray(highpass))
        lowpass = np.transpose(np.asarray(lowpass))


        print "finish convolution and adding up"
        print "highpass.shape: ", highpass.shape
        print "lowpass.shape: ", lowpass.shape

        (h, w) = highpass.shape

        # initialize arrays to hold the upsampled information
        highpass_up = []
        lowpass_up = []

        # upsample the columns by adding a column of zeros after every column
        # save information into initialized arrays
        for col in range(w):
            highpass_up.append(highpass[:, col])
            highpass_up.append(np.zeros(h))

            lowpass_up.append(lowpass[:, col])
            lowpass_up.append(np.zeros(h))

        # change to np.arrays and transpose
        highpass_up = np.transpose(np.asarray(highpass_up))
        lowpass_up = np.transpose(np.asarray(lowpass_up))

        print "finish up samlping"
        print "highpass_up.shape: ", highpass_up.shape
        print "lowpass_up.shape: ", lowpass_up.shape

        (h, w) = highpass_up.shape

        # initialize array to hold the original image
        # information extracted from the iDWT
        original_img = []

        # convolve the rows with the filter according to the
        # name of the array and sum the two arrays and save
        # information to original_img
        for row in range(h):
            convolution_result = np.convolve(highpass_up[row], self.h1) + np.convolve(lowpass_up[row], self.h0)
            original_img.append(convolution_result)
        
        # convert to np.array
        original_img = np.asarray(original_img)
        # original_img = original_img[:-3, :]
        print "finish convolution and adding up"
        print original_img.shape

        return original_img


    def dwt(self):
        """
        Run the 2-DWT (using Haar family) from the pywavelet library 
        on every tile and save coefficient results in tile object
        """
        # loop through the tiles
        for tile in self.tiles:
            cA, (cH, cV, cD)  = pywt.dwt2(tile.y_tile, 'haar') # library function returns tuple: (cA, (cH, cV, cD))
            tile.y_coeffs = (cA, cH, cV, cD) # save information as a 4-tuple, to mimic how our implementation saves the coeffs
            cA, (cH, cV, cD)  = pywt.dwt2(tile.Cb_tile, 'haar')
            tile.Cb_coeffs = (cA, cH, cV, cD)
            cA, (cH, cV, cD)  = pywt.dwt2(tile.Cr_tile, 'haar')
            tile.Cr_coeffs = (cA, cH, cV, cD)

            # cA, (cH, cV, cD) = tile.y_coeffs
            # print type(cA)
            # print cA.shape


        tile = self.tiles[0]
        print type(tile.y_coeffs[0])
        print tile.y_coeffs[0].shape
        cv2.imshow("tile.y_coeffs[0]", tile.y_coeffs[0])
        cv2.imshow("tile.y_coeffs[1]", tile.y_coeffs[1])
        cv2.imshow("tile.y_coeffs[2]", tile.y_coeffs[2])
        cv2.imshow("tile.y_coeffs[3]", tile.y_coeffs[3])

        cv2.waitKey(0)

    def idwt(self):
        """
        Run the inverse DWT (using the Haar family) from the pywavelet library
        on every tile and save the recovered tiles in the tile object
        """
        # loop through tiles
        for tile in self.tiles:
            if self.quant:
                # if tile was quantized, need to use the recovered, un-quantized coefficients
                tile.recovered_y_tile = pywt.idwt2(tile.recovered_y_coeffs, 'haar')  
                tile.recovered_Cb_tile = pywt.idwt2(tile.recovered_Cb_coeffs, 'haar')  
                tile.recovered_Cr_tile = pywt.idwt2(tile.recovered_Cr_coeffs, 'haar')  
            else:
                # if tile wasn't quantized, need to use the coeffs from DWT
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

    def quantization_math(self, img):
        """
        Quantize img: for every coefficient in img,
        save the original sign and decrease number of
        decimals saved by flooring the absolute value
        of the coeffcient divided by the step size
        """
        # initialize array to hold quantized coefficients,
        # to be same size as img
        (h, w) = img.shape
        quantization_img = np.empty_like(img)

        # loop through every coefficient in img
        for i in range(0, w):
            for j in range(0, h):
                # save the sign
                if img[j][i] >= 0:
                    sign = 1
                else:
                    sign = -1
                # save quantized coeffcicient
                quantization_img[j][i] = sign * math.floor(abs(img[j][i])/self.step)
        return quantization_img

    def i_quantization_math(self, img):
        """
        Inverse quantization of img: un-quantize
        the quantized coefficients in img by 
        multiplying the coeffs by the step size
        """
        # initialize array to hold un-quantized coefficients
        # to be same size as img
        (h, w) = img.shape
        i_quantization_img = np.empty_like(img)

        # loop through ever coefficient in img
        for i in range(0, w):
            for j in range(0, h):
                # save un-quantized coefficient
                i_quantization_img[j][i] = img[j][i] * self.step
        return i_quantization_img

    def quantization_helper(self, img):
        """
        Quantize the 4 different data arrays representing
        the 4 different coefficient approximations/details
        """
        cA = self.quantization_math(img[0])
        cH = self.quantization_math(img[1]) 
        cV = self.quantization_math(img[2]) 
        cD = self.quantization_math(img[3]) 
        
        return cA, cH, cV, cD

    def i_quantization_helper(self, img):
        """
        Un-quantize the 4 different data arrays representing
        the 4 different coefficient approximations/details
        """
        cA = self.i_quantization_math(img[0])
        cH = self.i_quantization_math(img[1]) 
        cV = self.i_quantization_math(img[2]) 
        cD = self.i_quantization_math(img[3]) 
        
        return cA, cH, cV, cD

    def quantization(self):
        """
        Quantize the tiles, saving the quantized
        information to the tile object
        """
        for tile in self.tiles:
            # quantize the tile in all 3 colorspaces
            tile.quantization_y = self.quantization_helper(tile.y_coeffs)
            tile.quantization_Cb = self.quantization_helper(tile.Cb_coeffs)
            tile.quantization_Cr = self.quantization_helper(tile.Cr_coeffs)

    def i_quantization(self):
        """
        Un-quantize the tiles, saving the un-quantized
        information to the tile object
        """
        for tile in self.tiles:
            tile.recovered_y_coeffs = self.i_quantization_helper(tile.quantization_y)
            tile.recovered_Cb_coeffs = self.i_quantization_helper(tile.quantization_Cb)
            tile.recovered_Cr_coeffs = self.i_quantization_helper(tile.quantization_Cr)

    def entropy_coding(self, img):
        # encode image
        pass

    def bit_stream_formation(self, img):
        # idk if we need this or what it is
        pass

    def forward(self):
        """
        Run the forward transformations to compress img
        """
        img = self.init_image(self.file_path)
        self.image_tiling(img)
        self.component_transformation()
        # self.dwt()
        self.DWT()
        if self.quant:
            self.quantization()

    def backward(self):
        """
        Run the backwards transformations to get the image back
        from the compressed data
        """
        if self.quant:
            self.i_quantization()
        # self.idwt()
        self.iDWT()
        self.i_component_transformation()

    def run(self):
        """
        Run forward and backward transformations, saving
        compressed image data and reconstructing the image
        from the compressed data
        """
        self.forward()
        self.backward()


if __name__ == '__main__':
    JPEG2000(tile_size = 900).run()
