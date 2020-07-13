# The Flatland Challenge

The [Flatland](https://pypi.org/project/flatland-rl/) environment has been designed for the purpose of testing and developing mult-agent reinforcement learning algorithims in the domain of [Vehical-Rescheduling-Problem (VSR)](https://en.wikipedia.org/wiki/Vehicle_rescheduling_problem), as was proposed by [Li, Mirchandani and Borenstein](https://onlinelibrary.wiley.com/doi/abs/10.1002/net.20199) in 2007. 

Flatland provides a simplistic grid world environment that simulates the dynamics of train traffic as well as the railway infrastructure. The goal is to develop a reinforcement learning based traffic management system that is able to select routes for all trains and decide on their priorities at switches in order to optimize traffic flow across the network.

As mentioned on Flatland's competition page 

``` The problems are formulated as a 2D grid environment with restricted transitions between neighboring cells to represent railway networks. On the 2D grid, multiple agents with different objectives must collaborate to maximize global reward. There is a range of tasks with increasing difficulty that need to be solved as explained in the coming sections.```

![flatland](ReadmeAndNotebookImages/flatland.gif)