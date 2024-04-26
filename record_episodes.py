import os
import cv2
import h5py
import argparse
import threading
from tqdm import tqdm
from time import sleep, time

from loki.config.config import ROBOT_PORTS, TASK_CONFIG
from loki.robot import Robot
from training.utils import pwm2pos, pwm2vel

# parse the task name via command line
parser = argparse.ArgumentParser()
parser.add_argument('--num_episodes', type=int, default=200)
args = parser.parse_args()
num_episodes = args.num_episodes

cfg = TASK_CONFIG
task = cfg['task_name']
frequency = cfg['frequency']
camera_queue = {}

def camera_thread_fn(cam_name, cam_idx):
    print(f"Starting camera thread for {cam_name} at port {cfg['camera_ports'][cam_idx]}")
    cam = cv2.VideoCapture(cfg['camera_ports'][cam_idx])
    if not cam.isOpened():
            raise IOError(f"Cannot open camera at port {cfg['camera_ports'][cam_idx]}")
    
    def capture_image():
        ret, frame = cam.read()
        if ret:
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            camera_queue[cam_name] = image
            # print(f"Captured image from {cam_name}")
        else:
            print(f"Failed to capture image from {cam_name}")
            
    while True:
        capture_image()

def get_images():
    images = {}
    for cam_name, image in camera_queue.items():
        images[cam_name] = image
    if images == {}:
        print('WARNING: no images captured')
    return images

def remove_last_episode():
    data_dir = os.path.join(cfg['dataset_dir'], task)
    idx = len([name for name in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, name))])
    idx -= 1
    dataset_path = os.path.join(data_dir, f'episode_{idx}')
    os.remove(dataset_path + '.hdf5')
    print(f'Removed last episode: {dataset_path}')

def save_episode_to_file(obs_replay, action_replay):
    # create a dictionary to store the data
    data_dict = {
        '/observations/qpos': [],
        '/observations/qvel': [],
        '/action': [],
    }
    # there may be more than one camera
    for cam_name in cfg['camera_names']:
            data_dict[f'/observations/images/{cam_name}'] = []

    # store the observations and actions
    for o, a in zip(obs_replay, action_replay):
        data_dict['/observations/qpos'].append(o['qpos'])
        data_dict['/observations/qvel'].append(o['qvel'])
        data_dict['/action'].append(a)
        # store the images
        for cam_name in cfg['camera_names']:
            data_dict[f'/observations/images/{cam_name}'].append(o['images'][cam_name])

    max_timesteps = len(data_dict['/observations/qpos'])
    # create data dir if it doesn't exist
    data_dir = os.path.join(cfg['dataset_dir'], task)
    if not os.path.exists(data_dir): os.makedirs(data_dir)
    # count number of files in the directory
    idx = len([name for name in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, name))])
    dataset_path = os.path.join(data_dir, f'episode_{idx}')
    # save the data
    with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
        root.attrs['sim'] = True
        obs = root.create_group('observations')
        image = obs.create_group('images')
        for cam_name in cfg['camera_names']:
            _ = image.create_dataset(cam_name, (max_timesteps, cfg['cam_height'], cfg['cam_width'], 3), dtype='uint8',
                                    chunks=(1, cfg['cam_height'], cfg['cam_width'], 3), )
        qpos = obs.create_dataset('qpos', (max_timesteps, cfg['state_dim']))
        qvel = obs.create_dataset('qvel', (max_timesteps, cfg['state_dim']))
        # image = obs.create_dataset("image", (max_timesteps, 240, 320, 3), dtype='uint8', chunks=(1, 240, 320, 3))
        action = root.create_dataset('action', (max_timesteps, cfg['action_dim']))
        
        for name, array in data_dict.items():
            root[name][...] = array
    
    print(f'Saved episode {idx} to {dataset_path}')

if __name__ == "__main__":
    # init camera
    cams = {}
    for i, cam_name in enumerate(cfg['camera_names']):
        threading.Thread(target=camera_thread_fn, args=(cam_name, i)).start()

    follower = Robot(device_name=ROBOT_PORTS['follower'], servo_ids=[1, 2, 3, 4, 5, 6, 7])
    leader = Robot(device_name=ROBOT_PORTS['leader'], servo_ids=[1, 2, 3, 4, 5, 6, 7])
    leader.set_trigger_torque()
    
    just_started = True
    for i in range(num_episodes):
        # reset to initial position (5 seconds)
        if just_started:
            just_started = False
            t0 = time()
            while time() - t0 < 5:
                a = leader.read_position(linear=True)
                follower.set_goal_pos(a)
                get_images()
        
        # if a[-1] < 1510:
        #     # if the episode starts with a closed gripper, remove the last episode
        #     remove_last_episode()
        #     continue
        
        # wait for images
        while len(camera_queue) < len(cfg['camera_names']):
            pass

        # os.system('say "go"')
        # init buffers
        obs_replay = []
        action_replay = []
        for i in tqdm(range(cfg['episode_len']), desc=f'Episode {i}'):
            st = time()
            # observation
            qpos = follower.read_position()
            qvel = follower.read_velocity()
            images = get_images()
            obs = {
                'qpos': pwm2pos(qpos),
                'qvel': pwm2vel(qvel),
                'images': images,
            }
            # action (leader's position)
            action = leader.read_position(linear=True)
            # apply action
            follower.set_goal_pos(action)
            action = pwm2pos(action)
            # store data
            obs_replay.append(obs)
            action_replay.append(action)
            sleep_t = max((1 / frequency) - (time() - st), 0)
            if sleep_t == 0:
                print('WARNING: frequency is too high')
            sleep(sleep_t)

        # os.system('say "stop"')
        threading.Thread(target=save_episode_to_file, args=(obs_replay, action_replay)).start()
        
    leader._disable_torque()
    # follower._disable_torque()
