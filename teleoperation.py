from loki.robot import Robot
from loki.config.config import ROBOT_PORTS

# init robots
leader = Robot(device_name=ROBOT_PORTS['leader'], servo_ids=[1, 2, 3, 4, 5, 6, 7])
follower = Robot(device_name=ROBOT_PORTS['follower'], servo_ids=[1, 2, 3, 4, 5, 6, 7])
# activate the leader gripper torque
leader.set_trigger_torque()

while True:
    follower.set_goal_pos(leader.read_position(linear=True))

#leader._disable_torque()
#follower._disable_torque()