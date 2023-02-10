from assignment_2_lib import take_a_photo, drive
import cv2
import numpy as np
import math

WIDTH = 640 # px
CENTER = WIDTH / 2
SCALING_FACTOR = 378 # tuned in the experimental way
EPSILON_1 = 30 # px
EPSILON_2 = 50 # px
DRIVE_STEPS = 250
FEW_STEPS_FORWARD = 10
FEW_STEPS_BACKWARD = 5

def mask_red(img):
  lower_red = np.array([0, 100, 50])
  upper_red = np.array([10, 255, 255])
  mask1 = cv2.inRange(img, lower_red, upper_red)
  lower_red = np.array([160, 100, 50])
  upper_red = np.array([179, 255, 255])
  mask2 = cv2.inRange(img, lower_red, upper_red)
  return mask1 + mask2

def mask_blue(img):
  lower_blue = np.array([100, 100, 50])
  upper_blue = np.array([140, 255, 255])
  mask = cv2.inRange(img, lower_blue, upper_blue)
  return mask

def mask_green(img):
  lower_green = np.array([40, 100, 50])
  upper_green = np.array([80, 255, 255])
  mask = cv2.inRange(img, lower_green, upper_green)
  return mask

def get_mask(photo, mask):
  img = photo[0:400, :, :]
  img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
  return mask(img)

def ball_diameter(mask):
  D = max(np.sum(mask, axis=1)) / 255
  return D

def forward_distance(photo):
  mask = get_mask(photo, mask_red)
  D = ball_diameter(mask)
  if D > 0:
    return round(SCALING_FACTOR * WIDTH / D)
  else:
    return math.inf

def ball_pos_in_cam(car):
  photo = take_a_photo(car)
  mask = get_mask(photo, mask_red)
  ball_x = np.argmax(np.sum(mask, axis=0))
  return ball_x

def gate_pos_in_cam(car, mask_func):
  photo = take_a_photo(car)
  mask = get_mask(photo, mask_func)
  indices = np.where(mask == 255)
  if not np.any(indices):
    gate_x = 0
  else:
    gate_x = np.mean(indices[1])
  return gate_x

def turn_to_object(car, obj_pos_in_cam):
  obj_x = obj_pos_in_cam(car)
  if obj_x < CENTER:
    direction = 1
  else:
    direction = -1
  forward = True
  while not (CENTER - EPSILON_1 <= obj_x <= CENTER + EPSILON_1):
    drive(car, forward, direction)
    forward = not forward
    direction = -direction
    obj_x = obj_pos_in_cam(car)

def find_a_ball(car):
  turn_to_object(car, ball_pos_in_cam)
  photo = take_a_photo(car)
  steps = forward_distance(photo)
  iterations = round(steps / DRIVE_STEPS)
  margin = math.ceil((30 - iterations) / 10)
  iterations -= margin
  for i in range(iterations):
    drive(car, True, 0)

def back_until_ball_visible(car):
  ball_x = ball_pos_in_cam(car)
  while ball_x == 0:
    drive(car, False, 0)
    ball_x = ball_pos_in_cam(car)

def turn_backwards(car, mask_func):
  ball_x = ball_pos_in_cam(car)
  gate_x = gate_pos_in_cam(car, mask_func)
  if ball_x < gate_x - EPSILON_2:
    for _ in range(FEW_STEPS_BACKWARD):
      drive(car, False, 1)
  elif ball_x > gate_x + EPSILON_2:
    for _ in range(FEW_STEPS_BACKWARD):
      drive(car, False, -1)

def back_until_ball_between_bars(car, mask_func):
  ball_x = ball_pos_in_cam(car)
  gate_x = gate_pos_in_cam(car, mask_func)
  while not (gate_x > 0 and ball_x > 0 and gate_x - EPSILON_2 <= ball_x <= gate_x + EPSILON_2):
    drive(car, False, 0)
    ball_x = ball_pos_in_cam(car)
    gate_x = gate_pos_in_cam(car, mask_func)

def forward_until_gate_not_visible(car, mask_func):
  gate_x = gate_pos_in_cam(car, mask_func)
  while EPSILON_1 <= gate_x <= WIDTH - EPSILON_1:
    drive(car, True, 0)
    gate_x = gate_pos_in_cam(car, mask_func)

def drive_into_the_gate(car, mask_func):
  gate_x = gate_pos_in_cam(car, mask_func)
  for _ in range(FEW_STEPS_FORWARD):
    if gate_x < CENTER:
      drive(car, True, 1)
    else:
      drive(car, True, -1)
    gate_x = gate_pos_in_cam(car, mask_func)

def move_a_ball(car):
  back_until_ball_visible(car)

  turn_backwards(car, mask_blue)
  back_until_ball_between_bars(car, mask_blue)
  find_a_ball(car)
  turn_to_object(car, ball_pos_in_cam)

  turn_backwards(car, mask_blue)
  back_until_ball_between_bars(car, mask_blue)
  find_a_ball(car)
  
  forward_until_gate_not_visible(car, mask_blue)
  drive_into_the_gate(car, mask_green)
