import numpy as np
import skimage.draw
from skimage.io import imread, imsave
from smu import mask_to_polygon, polygon_to_lists

# load image
path = "./blobs.png"
print("load image: %s" % path)
img = imread(path)
# get red channel
img = np.squeeze(img[:, :, 0])

# generate binary mask
print("generate mask")
mask = np.where(img == 255, 1, 0)

# detect blobs
print("detect blobs")
polys = mask_to_polygon(mask)
print("# blobs:", len(polys))

# convert polygon blobs into lists of coordinates and plot polygon perimeter
# turn blobs into dark gray ones
print("overlay polygons")
img = np.where(img == 255, 128, 0)
for poly in polys:
    # get coordinates
    px, py = polygon_to_lists(poly, swap_x_y=True, as_type="int")

    # draw polygon as white outline
    overlay = np.zeros(img.shape, dtype=np.uint8)
    rr, cc = skimage.draw.polygon_perimeter(py, px, shape=img.shape)
    overlay[rr, cc] = 255
    img = np.where(overlay == 255, 255, img)

# save image (requires matplotlib > 3.3)
path = "./blobs_poly.png"
print("save image: %s" % path)
img = img.astype(np.uint8)
imsave(path, img)
