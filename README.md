# ESD_NASim_Summer2022


Shreyas Branch

---------

## Goal 

To Create a streamlined way to automate and analyze the output by agents. This could be used in conjunction with the work of Abhay in environment changes or with Tejus and Liam for analysis of their created agents. The Automated output would contain least and most common action, type of and location of action, reward, number of steps in episode, and convergence.  

## Process

To get output from the agents I used the pandas library and added code to the train and other similar functions to write the output into a Pandas DataFrame. Within the main function I would then work on merging all the DataFrames together and analyzing that output. 

The first few iterations of the output are mainly in .ipynb files Only later did I start working on having it all in the main file. In the main function I worked to find correlation between variables and the least and most common action. I also analyzed the reward and typical amount of episodes for convergence on the small environment.


## Future Goals

Due to time constraints, there are some goals that I was unable to get to during the Summer Internship. I was unable to break the automated analysis into individual episodes and instead had to study it as a whole. A lot of the work for analyzing and studying the output is still done manually. Also it would be nice to see if this same process can be used for other non-vanilla agents or how the outputs would vary between network changes. 


## Other

I earlier made fixes to depreciated code within the Q learning agents. I believe these were later fixed in the actual main nasim repository. 

Also lots of the changes made are documented in the README.md within the nasim folder. 