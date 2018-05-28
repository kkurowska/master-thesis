import numpy as np
import matplotlib.pyplot as plt

import pywt
import pywt.data

wavelet_name = 'db1'

# Load image
original = pywt.data.camera()

# Wavelet transform of image, and plot approximation and details
titles = ['Approximation', ' Horizontal detail',
          'Vertical detail', 'Diagonal detail']

coeffs2 = pywt.dwt2(original, wavelet_name)
LL, (LH, HL, HH) = coeffs2
fig = plt.figure()
for i, a in enumerate([LL, LH, HL, HH]):
    ax = fig.add_subplot(2, 2, i + 1)
    ax.imshow(a, origin='image', interpolation="nearest", cmap=plt.cm.gray)
    ax.set_title(titles[i], fontsize=12)
fig.suptitle("dwt2 coefficients", fontsize=14)

coefs = [0, 0, 0, 0]
fig = plt.figure()
for i, a in enumerate([LL, LH, HL, HH]):
    da = pywt.threshold(a, 10, mode='hard') # some thresholding ?
    coefs[i] = da
    ax = fig.add_subplot(2, 2, i + 1)
    ax.imshow(da, origin='image', interpolation="nearest", cmap=plt.cm.gray)
    ax.set_title(titles[i], fontsize=12)
fig.suptitle("denoised coefficients", fontsize=14)

denoised_coeffs2 = coefs[0], (coefs[1], coefs[2], coefs[3])

# Now reconstruct and plot the original image
reconstructed = pywt.idwt2(denoised_coeffs2, wavelet_name)
fig = plt.figure()
plt.imshow(reconstructed, interpolation="nearest", cmap=plt.cm.gray)

plt.show()
