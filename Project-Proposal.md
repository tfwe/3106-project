# Proposal for AI Project: 

## AI Player for 2048 using Reinforcement Learning

- COMP 3106 A 
- Carlo Flores - 101156348
- Michael Macdougall - 101197828

## Introduction and Background
We would like to develop an AI player for the puzzle game 2048 using reinforcement learning techniques. 2048 is a puzzle game that involves shifting merging two pieces of equal value to reach a highest score, while mitigating the amount of space taken up in the grid by all of the tiles. It is a popular puzzle game while remaining relatively simple in principle.

## Proposed Objectives
- Develop an AI player that can successfully play the game 2048 to reach a maximum value tile of 2048 without losing the game.
- Alter the game of 2048 to have a powerup.
- Be able to evalute the worth of a given game state with the use of the added powerup.
- Implement reinforcement learning algorithms to train the AI player.
- Evaluate the performance of the AI player and compare its performance to its preceding iteration players.

## Possible Powerups
(Looking for feedback on which powerup would be the best fit)
- Negative tiles: these would be additional tiles that are negative, comnibining them with positive tiles would result in an empty space and combing them with other negative tiles will result in a larger negative tile. For example -2 and 2 would cancel and make an empty space and -2 and -2 would combine to a -4 tile.
- Random additon: This ability would be a button that adds a tile of random worth from the current tiles on the board. for example if the maximum tile on the board is 256 then tiles from 2 to 256 can be added. This powerup could be on cooldown or have a finite number of uses.
- Clear row/column: A button that would clear a select row or column of all its tiles with a finite amount of uses.

## Proposed Methods from Artificial Intelligence
AI example:
>We plan to use reinforcement learning techniques to train our AI player. Specifically, we will explore the use of Q-learning, a popular reinforcement learning algorithm, to teach the AI player how to make optimal moves in the game. We will also investigate the use of deep reinforcement learning, such as Deep Q-Networks (DQN), to further enhance the AI player's performance.

## Dataset
For this project, the AI player will only learn based on its own interactions with the 2048 game, so a dataset will not be needed.

## Proposed Validation/Analysis Strategy
To validate and analyze the performance of our AI player, we will compare its gameplay with that of its own previous iteration player. We can measure things like the highest score, number of moves moves made before losing, and the number of turns needed to reach certain milestones.
>continue?

## Description of Novelty
AI example:
>adding a "Swap" power-up? This power-up allows players to swap the positions of any two tiles on the board, helping them strategically rearrange the tiles to create better combinations and reach higher numbers. However, players would have a limited number of swaps available, adding a strategic element to the game.

list of possible novelty options

## Weekly Schedule and Milestones
AI example
>- Week 1: Familiarize ourselves with the game 2048 and its rules.
>- Week 2: Implement a basic version of the game environment and the AI player.
>- Week 3: Explore and implement Q-learning algorithm for training the AI player.
>- Week 4: Evaluate the performance of the AI player and make improvements if necessary.
>- Week 5: Investigate the use of deep reinforcement learning techniques to enhance the AI player's performance.
>- Week 6: Conduct experiments and analyze the results.
>- Week 7: Finalize the project and prepare for presentation.

## Projection Demonstration Time Slots
Carlo and I would prefer to do our demonstration in person and at any of these times:
>- Tuesday 12:00, 1:00
>- Thursday 12:00, 1:00
>- Friday 12:00, 1,00

## Gpu Requirements
> Since we're using deep learning a GPU would help train different iterations of our program simultaneously and so we would like access to a GPU in order to accelerate the training process.

