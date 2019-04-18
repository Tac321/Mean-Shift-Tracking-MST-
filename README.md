# MST: Mean Shift Tracking
Here shown is an (MST)  solution for tracking images. This is useful for low quality images of a discinct intensity/texture/color.


 
4
# MST: Mean Shift Tracking
5
Simulation environment of the Model Predictive Contouring Controller (MPCC) for Autonomous Racing developed by the Automatic Control Lab (IfA) at ETH Zurich
6
​
7
​
8
## How to run
9
​
10
### Before running code
11
1) Install ...
12
2) Use quadprog, "MPC_vars.interface = 'quadprog';" Note: use of Quadprog yeilds more accurate solution than my QP solver.
13
### Run code
14
1) run simulation.m
15
2) play with the tunning in getMPC_vars.m
16
3) change the car model between FullSize and ORCA
17
4) change the track layout between the ORCA and the RCP track
18
​
19
## Example
20
<img src="https://github.com/alexliniger/MPCC/blob/master/Images/MPC_sim.gif" width="700" />
21
​



