import numpy as np
import random
import cv2
from task1 import load_image
from task3 import find_homography
from task5 import stitch_images
from task6 import get_matches

def get_errors(matches, H):
  errors = []
  for match in matches:
    p1 = np.append(match[0], 1)
    p2 = np.append(match[1], 1)
    est_p2 = np.dot(H, p1)
    est_p2 /= est_p2[2]
    res = p2 - est_p2
    err = np.hypot(res[0], res[1])
    errors.append(err)
  return np.array(errors)

def ransac(matches, iterations, threshold, num_samples):
  max_inliers_count = 0
  best_H = None

  for i in range(iterations):
    samples = random.sample(matches, num_samples)
    H = find_homography(samples)

    if np.linalg.matrix_rank(H) < 3:
      continue

    errors = get_errors(matches, H)
    inliers_count = np.count_nonzero(errors < threshold)
    if inliers_count > max_inliers_count:
      max_inliers_count = inliers_count
      best_H = H
  return best_H

def main():
  img1 = load_image('images/undistorted/0.png')
  img2 = load_image('images/undistorted/1.png')
  matching_pts = get_matches('images/undistorted/0.png', 'images/undistorted/1.png', False)
  H = ransac(matching_pts, 10000, 10, 5)
  stitched = stitch_images(img1, img2, H)
  cv2.imwrite('images/panoramas/task7.png', stitched)

if __name__ == '__main__':
  main()
