# # import numpy as np 
# # # a = np.zeros((5,5))
# # # print a

# # # print a[0:3, 1:3]


# # component_transformation_matrix = np.array([[0.2999, 0.587, 0.114],
# #             [-0.16875, -0.33126, 0.5],[0.5, -0.41869, -0.08131]])

# # a = np.array([[1], [2], [3]])
# # print a.shape
# # print component_transformation_matrix.shape

# # print np.matmul(component_transformation_matrix, a)
# # # print component_transformation_matrix

# # print np.zeros((3,3))


# import numpy as np
# # import pywt
# # data = np.ones((4,4), dtype=np.float64)
# # coeffs = pywt.dwt2(data, 'haar')
# # cA, (cH, cV, cD) = coeffs

# # print cA

# a = np.ones((4, 4, 3))
# print np.zeros_like(a)

# # img = decoder.decode(openjpeg.IMAGE_EIT)

# import math

# print math.floor(0.5)

import numpy as np
import math


def quantization(img):
    step = 1.0/2
    (h, w, _) = img.shape
    quantization_img = np.empty_like(img)

    for i in range(0, w):    # for every pixel:
        for j in range(0, h):
            if img[j][i] >= 0:
                sign = 1
            else:
                sign = -1
            quantization_img[j][i] = sign * math.floor(abs(img[j][i])/step)
    return quantization_img

# a = np.zeros((4,4,1))
# a = np.array([[4, 4, 4], [4, 4, 4], [4, 4, 4], [4, 4, 4]])
b = np.random.randint(10, size=(4, 4, 1))

print b[0]
print quantization(b)[0]




