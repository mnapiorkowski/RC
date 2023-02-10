import numpy as np

def height_width(img):
  return img.shape[:2]

def new_coordinates(x, y, homography):
  new_vec = np.dot(homography, np.array([x, y, 1]))
  x_new = round(new_vec[0] / new_vec[2])
  y_new = round(new_vec[1] / new_vec[2])
  return x_new, y_new

def new_img_size(height, width, homography):
  up_left = new_coordinates(0, 0, homography)
  down_left = new_coordinates(0, height, homography)
  up_right = new_coordinates(width, 0, homography)
  down_right = new_coordinates(width, height, homography)
  leftest = min(up_left[0], down_left[0])
  rightest = max(up_right[0], down_right[0])
  uppest = min(up_right[1], up_left[1])
  downest = max(down_right[1], down_left[1])
  new_height = downest - uppest
  new_width = rightest - leftest
  shift = (-leftest, -uppest)
  return new_height, new_width, shift

def transform_image(src_img, homography):
  inv_homography = np.linalg.inv(homography)

  h_src, w_src = height_width(src_img)
  h_dst, w_dst, shift = new_img_size(h_src, w_src, homography)

  new_shape = list(src_img.shape)
  new_shape[:2] = h_dst, w_dst
  new_shape = tuple(new_shape)
  dst_img = np.zeros(new_shape)

  for y_dst in range(-shift[1], h_dst - shift[1]):
    for x_dst in range(-shift[0], w_dst - shift[0]):
      x_src, y_src = new_coordinates(x_dst, y_dst, inv_homography)
      if 0 <= x_src < w_src and 0 <= y_src < h_src:
        dst_img[y_dst + shift[1], x_dst + shift[0]] = src_img[y_src, x_src]

  return dst_img, shift
