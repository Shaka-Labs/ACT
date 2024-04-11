# Imiation Learning for 250$ robot arm
This repository contains a re-adapatation of [Action Chunking Transformer](https://github.com/tonyzhaozh/act/tree/main) that works for this [low-cost robot](https://github.com/AlexanderKoch-Koch/low_cost_robot) design (250$). 

We are sharing the repo so anyone (non-experts included) can train a robot policy after a few teleoperated demonstraions.

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

Go to `config/config.py` and change the paths of the ports that connect leader and follower robots to your computer.

### Data collection
In order to collect data simply run:
~~~
python record_episodes.py
~~~
You can select how many episodes you intend collecting in `config/config.py`. Change the value of `'num_episodes'` (set to 1 by default). You should be turning on the volume of your pc-- data for each episode will be recorded after you hear "Go" and it will stop when you hear "Stop".

You will also need to connect a camera to your computer and point it towards the robot while collecting the data via teleoperation. It's important the camera doesn't move otherwise evaluation of the policy is likely to fail.

### Train policy
We slightly re-adapt [Action Chunking Tranfosrmer](https://github.com/tonyzhaozh/act/tree/main) to account for our setup. To start training simply run:
~~~
python train.py
~~~
The policy will be saved in `checkpoints/`.

### Evaluate policy
Make sure to keep the same setup while you were collecting the data. To evaluate the policy simply run:
~~~
python evaluate.py
~~~