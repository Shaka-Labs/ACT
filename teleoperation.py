import cv2
import threading
from loki.robot import Robot
from loki.config.config import ROBOT_PORTS, TASK_CONFIG

camera_queue = {}
cfg = TASK_CONFIG

def camera_thread_fn(cam_name, cam_idx):
    print(f"Starting camera thread for {cam_name} at port {cfg['camera_ports'][cam_idx]}")
    cam = cv2.VideoCapture(cfg['camera_ports'][cam_idx])
    if not cam.isOpened():
            raise IOError(f"Cannot open camera at port {cfg['camera_ports'][cam_idx]}")
    
    def capture_image():
        ret, frame = cam.read()
        if ret:
            # image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            camera_queue[cam_name] = frame
        else:
            print(f"Failed to capture image from {cam_name}")
        
        # Show the image
        if cam_name == 'front':
            cv2.imshow(f'{cam_name}', frame)
            cv2.waitKey(1)
            
    while True:
        capture_image()

def get_images():
    images = {}
    for cam_name, image in camera_queue.items():
        images[cam_name] = image
    return images

# init robots
leader = Robot(device_name=ROBOT_PORTS['leader'], servo_ids=[1, 2, 3, 4, 5, 6, 7])
follower = Robot(device_name=ROBOT_PORTS['follower'], servo_ids=[1, 2, 3, 4, 5, 6, 7])
# activate the leader gripper torque
leader.set_trigger_torque()

# init camera
threading.Thread(target=camera_thread_fn, args=('front', 0)).start()
# threading.Thread(target=camera_thread_fn, args=('wrist', 1)).start()

while True:
    follower.set_goal_pos(leader.read_position(linear=True))

#leader._disable_torque()
#follower._disable_torque()