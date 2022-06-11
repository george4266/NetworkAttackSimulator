# ESD_NASim_Summer2022


Shreyas Branch


## Changes in Agents

For my branch, I deceided to analyze the results of agents within the nasim/agents directory. To do this I modified the agents by adding pandas in order to create DataFrames and .csv files for the outputs. 

So far there are three files that have had these modifications. As the list below goes on, there have been more modifications as I have learned more about the files and their structures. 
 - random_agent.py
 - bruteforce_agent.py
 - ql_agent.py


### Random and Brute Force Agents
---------
The Random and Brute Force are very similar in terms of structure. These have the least amount of analysis. In common both rewards decrease then suddenly increase before decreasing; a very jagged negative slopped pattern.

The brute force agents uses a .ipynb file to find the most common action taken. In the file I chose to save, the most common action list taken was used 63 times. I also found that the first 5 nodes within that most common path were used 100+ times. 

### Ql-Agent
-----

This agent had the most analysis. After the creation of the DataFrame I put the data into a google sheet. I then used google colab to generate correlation analysis and 3d scatter plots using plotly. 

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
