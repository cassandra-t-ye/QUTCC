from motionblur import Kernel
import imageio.v3 as iio
import numpy as np
import cv2

image = iio.imread('/share/monakhova/Cassandra_data/UQNet_proj/motionblur/mirflickr/im1.jpg')
image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_LINEAR)
print(image.shape, image.min(), image.max())
kernel = Kernel(size=(20, 20), intensity=0.1)
image = np.array(kernel.applyTo(image.copy(), keep_image_dim=True))
print(image.shape, image.min(), image.max())
iio.imwrite('output.jpg', image)
