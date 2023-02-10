import numpy as np
import cv2
from task1 import load_image
from task2 import transform_image, height_width
from task4 import get_H

def weights(img):
  h, w = height_width(img)
  weights = np.zeros((h, w))
  max_dist = max(w/2, h/2) # from the center

  for y in range(h):
    for x in range(w):
      dist = min(x, w - x, y, h - y) # to the closest edge
      weights[y, x] = dist**2 / max_dist
  return weights

def stitch_images(img1, img2, homography):
  weights_img1 = weights(img1)
  weights_img1, _ = transform_image(weights_img1, homography)
  weights_img2 = weights(img2)

  img1, shift = transform_image(img1, homography)
  h_img1, w_img1 = height_width(img1)
  h_img2, w_img2 = height_width(img2)
  res = []
  delta_h1, delta_w1, delta_h2, delta_w2 = 0, 0, 0, 0

  if shift[0] >= 0:
    if shift[1] >= 0:
      delta_h1 = 0
      delta_w1 = 0
      delta_h2 = shift[1]
      delta_w2 = shift[0]
    else:
      delta_h1 = -shift[1]
      delta_w1 = 0
      delta_h2 = 0
      delta_w2 = shift[0]
  else:
    if shift[1] >= 0:
      delta_h1 = 0
      delta_w1 = -shift[0]
      delta_h2 = shift[1]
      delta_w2 = 0
    else:
      delta_h1 = -shift[1]
      delta_w1 = -shift[0]
      delta_h2 = 0
      delta_w2 = 0

  res = np.zeros((max(h_img1 + delta_h1, h_img2 + delta_h2),
                  max(w_img1 + delta_w1, w_img2 + delta_w2), 3))
  weights_res = np.zeros(res.shape[:2])

  for y_img1 in range(h_img1):
    for x_img1 in range(w_img1):
      x_res = x_img1 + delta_w1
      y_res = y_img1 + delta_h1
      res[y_res, x_res] = img1[y_img1, x_img1]
      weights_res[y_res, x_res] = weights_img1[y_img1, x_img1]

  for y_img2 in range(h_img2):
    for x_img2 in range(w_img2):
      x_res = x_img2 + delta_w2
      y_res = y_img2 + delta_h2
      weight_img2 = weights_img2[y_img2, x_img2]
      weight_res = weights_res[y_res, x_res]
      if weight_res + weight_img2 != 0:
        res[y_res, x_res] = (weight_res * res[y_res, x_res] +
                            weight_img2 * img2[y_img2, x_img2]) / (weight_res + weight_img2)

  return res

def main():
  img1 = load_image('images/undistorted/0.png')
  img2 = load_image('images/undistorted/1.png')
  H_12 = get_H()
  stitched = stitch_images(img1, img2, H_12)
  cv2.imwrite('images/panoramas/task5.png', stitched)
  # stitched = stitch_images(img2, img1, H_21)
  # cv2.imwrite('images/panoramas/task5.png', stitched)

if __name__ == '__main__':
  main()
