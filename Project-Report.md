# Final Project - Solving 2048 with Monte Carlo Tree Search
**COMP 3106 A F23**

- Carlo Flores - 101156348
- Michael Macdougall - 101197828

## Statement of contributions
- Both Carlo and Michael made significant contributions
- Both Carlo and Michael made approximately equal contributions
- Carlo made contributions to  the implementation of the game 2048 as well as the implementation of Monte Carlo trees 

## Introduction
### Motivation behind implementation

### Objectives 

## Methods

### Q-Learning

### Deep Q-Learning (DQN)


### Monte Carlo Tree Search (MCTS)

## Results
We tested 3 different policies on 2048 to interpret and evaluate performance. To validate our results, we sampled 1000 games under a random move policy, and extracted the distribution of maximum and minimum value tiles, the number of turns, and the distribution of different values pieces as each game progressed. We used each of these inferences as our ground truths, and used them to compare the results we obtained from our DQN and MCTS policies. For visual simplicity, we are using the base 2 logarithm of each piece's respective value to represent it.

### Random Sample Games:
![random_1000_max_tile](https://github.com/tfwe/3106-project/assets/93735375/8eb3f3ac-c2a8-45cc-b66d-7cb79489f9a7)
![random_1000_tile_dist](https://github.com/tfwe/3106-project/assets/93735375/7360e8db-be8b-4d32-9ea4-473b6ff1e8cb)
![random_1000_num_turns](https://github.com/tfwe/3106-project/assets/93735375/1e6300cc-97d8-4f5b-bd6d-178cced67c6e)
![random_1000_min_tile](https://github.com/tfwe/3106-project/assets/93735375/26e9ba0f-47c1-4a97-90b1-9524c5e457fc)

### DQN Sample Games:
![qnn_1000_tile_dist](https://github.com/tfwe/3106-project/assets/93735375/c8a8a054-38db-4f22-a2b5-d0edd819d131)
![qnn_1000_num_turns](https://github.com/tfwe/3106-project/assets/93735375/1b53383e-3d36-4ee3-9546-b6f80ff71be4)
![qnn_1000_min_tile](https://github.com/tfwe/3106-project/assets/93735375/8f37256b-5958-4256-b4c3-770f796d6bb3)
![qnn_1000_max_tile](https://github.com/tfwe/3106-project/assets/93735375/0e0089e0-7722-433e-b088-84591a557fe2)

### MCTS Sample Games:
![mcts_1000_tile_dist](https://github.com/tfwe/3106-project/assets/93735375/a97b2fd0-582b-43f1-8505-43baf1cd5c5f)
![mcts_1000_num_turns](https://github.com/tfwe/3106-project/assets/93735375/2412efe3-420c-4c9a-ae55-f8a6018d564a)
![mcts_1000_min_tile](https://github.com/tfwe/3106-project/assets/93735375/fa931b5f-28ec-4dda-a4dd-d4a298832afd)
![mcts_1000_max_tile](https://github.com/tfwe/3106-project/assets/93735375/cd052f49-8d9c-43a0-84ca-c29774bc65e8)


## Discussion

## Running the Project Locally
This project can be run by cloning or downloading the repository on GitHub (https://github.com/tfwe/3106-project/) and running the following shell commands in the root directory of the downloaded repository. 

  >`pip3 install -r requiremnets.txt`
  
  >`python3 test.py [NUM_RUNS]`

If specified, `NUM_RUNS` will allow `test.py` to sample `NUM_RUNS` games from each policy. For example, we can run 

  >`python3 test.py 1000`

to extract data from 1000 games on each policy. 

## References

GitHub. (n.d.). https://github.com/qwert12500/2048_rl/blob/main/2048_Reinforcement_Learning.ipynb 

Silver, D., Hubert, T., Schrittwieser, J., Antonoglou, I., Lai, M., Guez, A., Lanctot, M., Sifre, L., Kumaran, D., Graepel, T., Lillicrap, T., Simonyan, K., & Hassabis, D. (2017). Mastering chess and shogi by self-play with a general reinforcement learning algorithm. https://doi.org/10.48550/ARXIV.1712.01815
