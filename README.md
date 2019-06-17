# Using Reinforcement Learning to Learn How To Play Doodle Jump

## Overview
This project follows the description of the Deep Q Learning algorithm described in Playing Atari with Deep Reinforcement Learning [1] and shows that this learning algorithm can be further generalized to the doodle jump.

Furthermore, I implement Deep Recurrent Q Learning [2] and outperform the original DQN a lot.  
## Installation Dependencies:
* Python 3
* TensorFlow 
* pygame
* OpenCV-Python
## Experiment
* Since deep Q-network is trained on the raw pixel values observed from the game screen at each time step. I remove the background appeared in the original game to make it converge faster.

## Preprocessing
* I resize to 80x80, gray scaling, normalize the input image and stack 4 input images as a input tensor.
## args
* -\-actions 
    2: actions contains [LEFT, RIGHT] 
    3: actions contains [LEFT, RIGHT, DO_NOTHING]

* -\- exp_dir
    the folder that contains model weights.
    
### args only for TESTING
* -\-iterations  
        testing in iterations and it count by death.
* -\-fps
        for visualizing the testing process, it will slow down the inferencing so not recommend for recoding benchmark.
        
## Training DQN
```shell
python train_with_dqn.py --actions 2 --exp_dir exp_dqn_1
```

## Training DRQN
```shell
python train_with_drqn.py --actions 2 --exp_dir exp_drqn_1
```

## Testing DQN
* `--fps 60` if you want to visualize the process, else remove `--fps 60`.
```shell
python test_with_dqn.py --actions 2 --exp_dir exp_drqn --fps 60
```

## Testing DRQN
* `--fps 60` if you want to visualize the process, else remove `--fps 60`.
```shell
python test_with_drqn.py --actions 2 --exp_dir exp_drqn --fps 60
```

## Result
* DRQN: trained 1285,0000 iterations
* DQN : trained 1605,0000 iterations
* test until 100 death.

_                         | DQN            | DRQN
--------                  | :-----------:  | :-----------: 
Survived time per death(s)|0.77            |10.3
top score per death       |17              |430
avg score in 100 death    |3               |96
           

## References
[1] Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Alex Graves, Ioannis Antonoglou, Daan Wierstra, and Martin Riedmiller. **Playing Atari with Deep Reinforcement Learning**. NIPS, Deep Learning workshop
[2] Matthew Hausknecht and Peter Stone. **Deep Recurrent Q-Learning for Partially Observable MDPs**. AAAI 2015
[3] https://github.com/f-prime/DoodleJump
