# import numpy as np 
# # a = np.zeros((5,5))
# # print a

# # print a[0:3, 1:3]


# component_transformation_matrix = np.array([[0.2999, 0.587, 0.114],
#             [-0.16875, -0.33126, 0.5],[0.5, -0.41869, -0.08131]])

# a = np.array([[1], [2], [3]])
# print a.shape
# print component_transformation_matrix.shape

# print np.matmul(component_transformation_matrix, a)
# # print component_transformation_matrix

# print np.zeros((3,3))


import numpy as np
# import pywt
# data = np.ones((4,4), dtype=np.float64)
# coeffs = pywt.dwt2(data, 'haar')
# cA, (cH, cV, cD) = coeffs

# print cA

a = np.ones((4, 4, 3))
print np.zeros_like(a)