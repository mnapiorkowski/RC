import numpy as np
import random
import cv2
from task2 import new_coordinates

def find_homography(matching_pts):
  A = []
  for match in matching_pts:
    p1 = np.append(match[0], 1)
    p2 = match[1]
    A.append([0, 0, 0, p1[0], p1[1], p1[2], -p2[1]*p1[0], -p2[1]*p1[1], -p2[1]*p1[2]])
    A.append([p1[0], p1[1], p1[2], 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1], -p2[0]*p1[2]])

  _, _, V = np.linalg.svd(np.array(A))
  eigenvector = V[-1, :]
  H = eigenvector.reshape(3, 3)
  H /= H[2, 2]
  return H

def random_point():
  x = random.randint(0, 1000)
  y = random.randint(0, 1000)
  return (x, y)

def random_homography():
  pts_src = []
  pts_dst = []
  for _ in range(4):
    x_src, y_src = random_point()
    x_dst, y_dst = random_point()
    pts_src.append((x_src, y_src))
    pts_dst.append((x_dst, y_dst))
  H, _ = cv2.findHomography(np.array(pts_src), np.array(pts_dst))
  return H

def test():
  passed = 0
  iterations = 100
  for _ in range(iterations):
    H = random_homography()
    matches = []
    for _ in range(1000):
      x_src, y_src = random_point()
      x_dst, y_dst = new_coordinates(x_src, y_src, H)
      matches.append(((x_src, y_src), (x_dst, y_dst)))
    H_new = find_homography(matches)
    diff = abs(H_new - H)
    same = not np.any(diff > 0.1)
    if same:
      passed += 1
  print(f'{passed}/{iterations} tests passed')

if __name__ == '__main__':
  test()
