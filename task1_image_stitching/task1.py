import numpy as np
import cv2

camera_matrix = np.array([[769.06214726, 0., 404.6773241],
                          [0., 768.16987645, 301.26633327],
                          [0., 0., 1.]])

dist_coeffs = np.array([1.07620348e-01, -3.03965753e-01, 4.39227698e-04, -4.64832562e-05, 1.79444432e-01])

def load_image(img_name):
  img = cv2.imread(img_name)
  return img

def undistort_image(img, alpha = 0):
  img_size = (img.shape[1], img.shape[0])
  rect_camera_matrix = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, img_size, alpha)[0]
  map1, map2 = cv2.initUndistortRectifyMap(camera_matrix, dist_coeffs, np.eye(3), rect_camera_matrix, img_size, cv2.CV_32FC1)
  rect_img = cv2.remap(img, map1, map2, cv2.INTER_LINEAR)
  return rect_img

def main():
  img1 = load_image('images/captured/0.png')
  img2 = load_image('images/captured/1.png')

  img1 = undistort_image(img1)
  img2 = undistort_image(img2)

  cv2.imwrite('images/undistorted/0.png', img1)
  cv2.imwrite('images/undistorted/1.png', img2)

if __name__ == '__main__':
  main()
