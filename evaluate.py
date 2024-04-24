from loki.config.config import POLICY_CONFIG, TASK_CONFIG, TRAIN_CONFIG, ROBOT_PORTS # must import first

import os
import cv2
import torch
import pickle
import argparse
import threading
from time import time, sleep

from loki.robot import Robot
from training.utils import *

# parse the task name via command line
parser = argparse.ArgumentParser()
args = parser.parse_args()

# config
cfg = TASK_CONFIG
task = cfg['task_name']
policy_config = POLICY_CONFIG
train_cfg = TRAIN_CONFIG
device = os.environ['DEVICE']
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
        
        # Show the image
        cv2.imshow('image', image)
        cv2.waitKey(1)
            
    while True:
        capture_image()

def get_images():
    images = {}
    for cam_name, image in camera_queue.items():
        images[cam_name] = image
    return images

if __name__ == "__main__":
    # init camera
    cams = {}
    for i, cam_name in enumerate(cfg['camera_names']):
        threading.Thread(target=camera_thread_fn, args=(cam_name, i)).start()
    sleep(1)
    # init follower
    follower = Robot(device_name=ROBOT_PORTS['follower'], servo_ids=[1, 2, 3, 4, 5, 6, 7], disable_torque=False)
    pos = follower.read_position()
    
    # Go to the initial position
    initial_pos = np.array([2080, 1832, 2029, 2036, 1187, 2045, 2866])
    for _ in range(100):
        follower.set_goal_pos(initial_pos)
        sleep(0.01)

    # load the policy
    ckpt_path = os.path.join(train_cfg['checkpoint_dir'], train_cfg['eval_ckpt_name'])
    policy = make_policy(policy_config['policy_class'], policy_config)
    loading_status = policy.load_state_dict(torch.load(ckpt_path, map_location=torch.device(device)))
    print(loading_status)
    policy.to(device)
    policy.eval()

    print(f'Loaded: {ckpt_path}')
    stats_path = os.path.join(train_cfg['checkpoint_dir'], f'dataset_stats.pkl')
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)

    pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
    post_process = lambda a: a * stats['action_std'] + stats['action_mean']

    query_frequency = policy_config['num_queries']
    if policy_config['temporal_agg']:
        query_frequency = 1
        num_queries = policy_config['num_queries']
    
    obs = {
        'qpos': pwm2pos(follower.read_position()),
        'qvel': vel2pwm(follower.read_velocity()),
        'images': get_images()
    }
    os.system('say "start"')

    n_rollouts = 1
    for i in range(n_rollouts):
        ### evaluation loop
        if policy_config['temporal_agg']:
            all_time_actions = torch.zeros([cfg['episode_len'], cfg['episode_len']+num_queries, cfg['state_dim']]).to(device)
        # qpos_history = torch.zeros((1, cfg['episode_len'], cfg['state_dim'])).to(device)
        with torch.inference_mode():
             # init buffers
            obs_replay = []
            action_replay = []
            # for t in range(cfg['episode_len']):
            t = -1
            while True:
                st = time()
                t += 1
                qpos_numpy = np.array(obs['qpos'])
                qpos = pre_process(qpos_numpy)
                qpos = torch.from_numpy(qpos).float().to(device).unsqueeze(0)
                # qpos_history[:, t] = qpos
                images = get_images()
                # TODO: change this for multiple cameras
                camera_names = cfg['camera_names']
                img = get_image(images, camera_names)

                # raw_action = policy(qpos, img)[:, 0]
                
                # if t % query_frequency == 0:
                APPLIED_HORIZON_LENGTH = 100
                if t % APPLIED_HORIZON_LENGTH == 0:
                    all_actions = policy(qpos, img)
                if policy_config['temporal_agg']:
                    all_time_actions[[t], t:t+num_queries] = all_actions
                    # all_time_actions[[t], t:t+APPLIED_HORIZON_LENGTH] = all_actions[:, :APPLIED_HORIZON_LENGTH]
                    actions_for_curr_step = all_time_actions[:, t]
                    actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                    actions_for_curr_step = actions_for_curr_step[actions_populated]
                    k = 0.01
                    exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                    exp_weights = exp_weights / exp_weights.sum()
                    exp_weights = torch.from_numpy(exp_weights.astype(np.float32)).to(device).unsqueeze(dim=1)
                    raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                else:
                    raw_action = all_actions[:, t % APPLIED_HORIZON_LENGTH]

                ### post-process actions
                raw_action = raw_action.squeeze(0).cpu().numpy()
                action = post_process(raw_action)
                action = pos2pwm(action).astype(int)
                ### take action
                follower.set_goal_pos(action)

                ### update obs
                obs = {
                    'qpos': pwm2pos(follower.read_position()),
                    'qvel': vel2pwm(follower.read_velocity()),
                    'images': images
                }
                ### store data
                obs_replay.append(obs)
                action_replay.append(action)

                sleep_t = max((1 / frequency) - (time() - st), 0)
                if sleep_t == 0:
                    print('WARNING: frequency is too high')
                sleep(sleep_t)
