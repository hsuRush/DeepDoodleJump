# Using Reinforcement Learning to Learn How To Play Doodle Jump
![demo gif](https://github.com/hsuRush/DeepDoodleJump/blob/master/readme_images/success_drqn.gif?raw=true)
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
* Also, To make the network train faster, I close off the red and blue platforms with only green platforms remain.

## Preprocessing

![preprocessing](https://github.com/hsuRush/DeepDoodleJump/blob/master/readme_images/preprocessing.png?raw=true)

## Terminated Problem (Non-stop)
![problem](https://github.com/hsuRush/DeepDoodleJump/blob/master/readme_images/problem.gif?raw=true)
* Solution: The game will be terminated if there's no veritcal camera movement in over 5 seconds.

## DQN Architecture
* I Stack 4 input images as a input tensor
![DQN network](https://github.com/hsuRush/DeepDoodleJump/blob/master/readme_images/DQN_network.png?raw=true)
## DRQN Architecture
* I Stack only **1** input images as a input tensor
![DQN network](https://github.com/hsuRush/DeepDoodleJump/blob/master/readme_images/DRQN_network.png?raw=true)
* replace a Dense layer as a **LSTM** layer.

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
Survived time per death(s)|0.77            |**10.3**
top score per death       |17              |**430**
avg score in 100 death    |3               |**96**

-------------

Updated (6/18)

* DRQN: trained **1960,0000** iterations
* DQN : trained 1605,0000 iterations
* test until 100 death.

_                         | DQN            | DRQN
--------                  | :-----------:  | :-----------: 
Survived time per death(s)|0.77            |**22.79**
top score per death       |17              |**1464**
avg score in 100 death    |3               |**256**
## Conclusion

![reward](https://github.com/hsuRush/DeepDoodleJump/blob/master/readme_images/reward.png?raw=true)
* The hidden reward platform (①,②) to the real reward(③) can be seen as a time sequence reward, the RNN is great at solving time sequence issues.
## References
[1] Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Alex Graves, Ioannis Antonoglou, Daan Wierstra, and Martin Riedmiller. **Playing Atari with Deep Reinforcement Learning**. NIPS, Deep Learning workshop

[2] Matthew Hausknecht and Peter Stone. **Deep Recurrent Q-Learning for Partially Observable MDPs**. AAAI 2015

[3] https://github.com/f-prime/DoodleJump
