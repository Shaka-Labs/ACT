# Imitation Learning for 250$ robot arm
This repository contains a re-adapatation of [Action Chunking Transformer](https://github.com/tonyzhaozh/act/tree/main) that works for this [low-cost robot](https://github.com/AlexanderKoch-Koch/low_cost_robot) design (250$). 

We are sharing the repo so anyone (non-experts included) can train a robot policy after a few teleoperated demonstraions.

The sorting task in the video was trained with less than 30 demonstrations on an RTX 3080 and took less than 30min.

https://github.com/Shaka-Labs/ACT/assets/45405956/83c05915-7442-49a4-905a-273fe35e84ee


## AI training
### Setup
Create conda environment
~~~
conda create --name act python=3.9
conda activate act
~~~

Install torch (for reference we add the versions we are using)
~~~
conda install pytorch==1.13.1 torchvision==0.14.1
~~~

You can now install the requirements:
~~~
pip install -r requirements.txt
~~~

Go to `TASK_CONFIG` in `config/config.py` and change the paths of the ports that connect leader and follower robots to your computer. 

You will also need to connect a camera to your computer and point it towards the robot while collecting the data via teleoperation. You can change the camera port in the config (set to 0 by default). It's important the camera doesn't move otherwise evaluation of the policy is likely to fail. 

### Data collection
In order to collect data simply run:
~~~
python record_episodes.py --task sort
~~~
You can define the name of the task you are doing and the episodes will be stored at `data/<task>`. You can also select how many episodes to collect when running the script by passing the argument `--num_episodes 1` (set to 1 by default). After getting a hold of it you can easily do 20 tasks in a row.

Turn on the volume of your pc-- data for each episode will be recorded after you hear "Go" and it will stop when you hear "Stop".

### Train policy
We slightly re-adapt [Action Chunking Tranfosrmer](https://github.com/tonyzhaozh/act/tree/main) to account for our setup. To start training simply run:
~~~
python train.py --task sort
~~~
The policy will be saved in `checkpoints/<task>`.

### Evaluate policy
Make sure to keep the same setup while you were collecting the data. To evaluate the policy simply run:
~~~
python evaluate.py --task sort
~~~
