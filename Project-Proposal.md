- Carlo Flores - 101156348
- Michael Macdougall - 101197828
## Introduction and Background
We would like to develop an AI player for the puzzle game 2048 using reinforcement learning techniques. 2048 is a puzzle game that involves shifting merging two pieces of equal value to reach a highest score, while mitigating the amount of space taken up in the grid by all of the tiles. It is a popular puzzle game while remaining relatively simple in principle.
## Proposed Objectives
- Alter the game of 2048 to change the common strategies used to succeed.
- Evaluate a score of a given game state with the use of the added modification.
- Implement reinforcement learning algorithms to train the AI player.
- Evaluate the performance of the AI player and compare its performance to its preceding iteration players.
## Proposed Methods from Artificial Intelligence
We would like to use reinforcement learning techniques to train our AI player. Specifically, we would like to explore the use of Q-learning to teach the AI player how to make optimal moves in the game. We would also like to explore strategies to optimize the agent's ability to learn.
## Dataset
For this project, the AI player will only learn based on its own interactions with the 2048 game, so a dataset will not be needed.
## Proposed Validation/Analysis Strategy
To validate and analyze the performance of our AI player, we will compare its gameplay with that of its own previous iteration player. We can measure things like the highest score, number of moves moves made before losing, and the number of turns needed to reach certain milestones. We can also look at how our AI player values each possible move in the current game state. The AI should select a value for each possible move it can make and choose the most efficient method whether that's using an availible powerup or just using the regular game mechanics to get to the next best possible state it can.
## Description of Novelty
In order to introduce novelty to our project in comparison to other similar projects done by others in the past, we would like to attempt experiments on our agent's performance when the fundamental rules of the game are changed. For example, we could introduce a 5th move that allows for a special action to occur in the game.
**Possible Modifications to Game for Novelty**
- Increasing or decreasing the size of the board
- An obstacle that can block the path of merging tiles
- Swap ability: An ability to swap two pieces on the board with a cooldown
- Negative tiles: these would be additional tiles that are negative, comnibining them with positive tiles would result in an empty space and combing them with other negative tiles will result in a larger negative tile. For example -2 and 2 would cancel and make an empty space and -2 and -2 would combine to a -4 tile.
- Random additon: This ability would be a button that adds a tile of random worth from the current tiles on the board. for example if the maximum tile on the board is 256 then tiles from 2 to 256 can be added. This powerup could be on cooldown or have a finite number of uses.
- Clear row/column: A button that would clear a select row or column of all its tiles with a finite amount of uses.
## Weekly Schedule and Milestones
>- Week 1: Implement a basic version of the game environment and the AI player.
>- Week 2: Explore and implement Q-learning algorithm for training the AI player.
>- Week 3: Evaluate the performance of the AI player and make improvements if necessary.
>- Week 4: Investigate the use of deep reinforcement learning techniques to enhance the AI player's performance.
>- Week 5: Conduct experiments and analyze the results.
>- Week 6: Finalize the project and prepare for presentation.
## Projection Demonstration Time Slots
Carlo and I would prefer to do our demonstration in person and at any of these times:
>- Tuesday 12:00, 1:00
>- Thursday 12:00, 1:00
>- Friday 12:00, 1,00
