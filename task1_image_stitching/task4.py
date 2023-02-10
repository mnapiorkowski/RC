from task3 import find_homography

def get_H():
  matching_pts = [((492, 93), (130, 78)),
                  ((482, 495), (124, 506)),
                  ((576, 161), (223, 155)),
                  ((698, 349), (344, 340)),
                  ((513, 286), (160, 282))]

  H_12 = find_homography(matching_pts)
  # H_21 = find_homography([(m[1], m[0]) for m in matching_pts])
  return H_12
