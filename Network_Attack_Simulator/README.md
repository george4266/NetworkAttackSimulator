# ESD_NASim_Summer2022


Shreyas Branch


## Changes in Agents

For my branch, I deceided to analyze the results of agents within the nasim/agents directory. To do this I modified the agents by adding pandas in order to create DataFrames and .csv files for the outputs. 

So far there are three files that have had these modifications. As the list below goes on, there have been more modifications as I have learned more about the files and their structures. 
 - random_agent.py
 - bruteforce_agent.py
 - ql_agent.py
 - ql_replay_agent.py


### Random and Brute Force Agents
---------
The Random and Brute Force are very similar in terms of structure. These have the least amount of analysis. In common both rewards decrease then suddenly increase before decreasing; a very jagged negative slopped pattern.

The brute force agents uses a .ipynb file to find the most common action taken. In the file I chose to save, the most common action list taken was used 63 times. I also found that the first 5 nodes within that most common path were used 100+ times. 

### Ql-Agent
-----

This agent had the most analysis. After the creation of the DataFrame I put the data into a google sheet. I then used google colab to generate correlation analysis and 3d scatter plots using plotly. 

## Running to Convergence

The Ql-agent had modifications at the argparser parts of the file in order to run until convergence. Link to that below:
https://docs.google.com/presentation/d/19eq2pHNGtTJ7_RPbyYcsolIyzGA5ClMpNC-lz7pfYq0/edit#slide=id.g134b413c3ff_0_0 


----
### QL Replay Agent

Agent was run until convergence and DataFrames were created. 

-----

## Changes to .yaml files 

### /saved_out 
This is just a folder containing .csv output files I chose to save for analysis.

------

### /scenarios/benchmark/my_network.yaml
Added a "my_network.yaml" file to play around with agents in my own custom environment. It was kinda neat but it takes so long to make a .yaml from scratch :P 

------

### /scenarios/benchmark/yaml_test and yaml_mod
An attempt to make a class file to create changes to any yaml file. I found this to be too  repetitive so I didn't invest more time into it... 

-----
## IPYNB folder

In order to better analyze the output from the RL agents, I have a subfolder within the nasim/agents folder. These python notebooks are to give visual aid and correlation analysis to the outputted .csv files. This means that the analysis does not have to be within the agent file as was the case when I made my original modifications the the ql_agent.py file. 

------
## Intereesting Files to study. 

> NetworkAttackSimulator/nasim/scenarios/__init__.py 
The nasim init file. This contains the functions to create benchmarks, generate and load scenarios, and get scenario max. 

> NetworkAttackSimulator/nasim/scenarios/generator.py
As stated in the file: "This file generates network configurations and action space configurations based of the hosts and services in network using standard formula". 

The exploits are all based on probabilities. The probabilities depend on what is wanted by the agent. These are explained after the class definition. Then follows the code:
```
def generate(self,
                 num_hosts,
                 num_services,
                 num_os=2,
                 ...
                 address_space_bounds=None,
                 **kwargs):
```


>NetworkAttackSimulator/nasim/agents/ql_agent.py

```class TabularQFunction:```

This class is specifically for the regualr ql agent. 



