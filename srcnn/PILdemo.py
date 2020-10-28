from PIL import Image
import matplotlib.pyplot as plt

img = Image.open("../test_data/B101/000t1.bmp")

plt.figure("1")
plt.imshow(img)
plt.show()