# Tasks

Stubs for tasks are available in the `assignment_2_solution.py`

Semi-automatic grading will be done using `assignment_2_tests.py`

There are three tasks. Your submission should be only one file, a modified version of `assignment_2_solution.py`. You should not read any actual data from pybullet (i.e. do not read positions and orientations of objects), you should decide how to steer using only photos from the camera.

## Task 1

The first task is to estimate how long (number of simulation steps) the car should drive forward to be near the target. The target (red ball) should not move.

To be specific: at the end of the movement car should be less then 1 m from the center of the ball. Ball should not move more than 10 cm, as in the code below:

```python
car_ball = distance.euclidean(p.getBasePositionAndOrientation(car)[0],
          p.getBasePositionAndOrientation(ball)[0])
assert car_ball < 1, car_ball

ball_move = distance.euclidean(ball_start_pos, p.getBasePositionAndOrientation(ball)[0])
assert ball_move < 0.1, ball_move
```

You should write a function that takes an image as an input and returns number of simulation steps the car should drive forward in order to reach the ball.

```python
def forward_distance(photo):
  #TODO: magic
  return some_value
```

Rules:

- Do not read coordinates of objects from the simulator.
- The only function you should change is `forward_distance`.
- The red ball is placed randomly on the line segment from `[2, 0, 0]` to `[5, 0, 0]`.
- The ball should not move.
- Two asserts in `test_forward_distance` check if the requirements are fulfilled.

Hint: Experiment with different ideas of how to calculate forward_distance. See what works, tune the parameters.

## Task 2

During the second task the ball is placed randomly in `[-3, -3] x [3,3]` square, but not too close to `[0, 0, 0]` (more then 1 m away).

Your job is to find the ball and drive close to it without moving it.

You should write `find_a_ball` function that has a loop, in which it takes photos and moves the car accordingly.

Rules:

- Do not read coordinates of objects from the simulator.
- The ball should not move.
- Use `take_a_photo(car)` function for making photos.
- Use `drive(car, forward, direction)` function for driving (forward/backward, left/straight/right).

## Task 3

This time you have to move the ball through the gate. The ball is randomly positioned within the rectangular region defined by the coordinates `[1, -1]` and `[2, 1]`. The car begins at a random point on the line defined by the coordinates `[4, -1]` and `[4, 1]` facing the negative x direction, allowing the gate to be seen by the camera. The gate is comprised of two large blue cylinders at the points `[-2, -1]` and `[-2, 1]`. There and two additional green cylinders at the points `[-4, -1]` and `[-4, 1]`.

The bumper is designed so that the ball does not come loose from it.

It's enough when at the end of `move_a_ball(pos)` the ball is located in `[-4, -1] x [-2, 1]` rectangle.

In the `move_a_ball` function you should (as in Task 2):

- not read coordinates of objects from simulator
- use `take_a_photo(car)` function for making photos
- use `drive(car, forward, direction)` function for driving (forward/backward, left/straight/right)
