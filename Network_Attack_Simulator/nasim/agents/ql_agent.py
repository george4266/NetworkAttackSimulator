"""An example Tabular, epsilon greedy Q-Learning Agent.
This agent does not use an Experience replay (see the 'ql_replay_agent.py')
It uses pytorch 1.5+ tensorboard library for logging (HINT: these dependencies
can be installed by running pip install nasim[dqn])
To run 'tiny' benchmark scenario with default settings, run the following from
the nasim/agents dir:
$ python ql_agent.py tiny
To see detailed results using tensorboard:
$ tensorboard --logdir runs/
To see available hyperparameters:
$ python ql_agent.py --help
Notes
-----
This is by no means a state of the art implementation of Tabular Q-Learning.
It is designed to be an example implementation that can be used as a reference
for building your own agents and for simple experimental comparisons.
"""
import random
import numpy as np
from pprint import pprint
import pandas as pd 
import plotly.express as px
import plotly.graph_objects as go
from rich import print as rprint








import nasim

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError as e:
    from gym import error
    raise error.DependencyNotInstalled(
        f"{e}. (HINT: you can install tabular_q_learning_agent dependencies "
        "by running 'pip install nasim[dqn]'.)"
    )


class TabularQFunction:
    """Tabular Q-Function """

    def __init__(self, num_actions):
        self.q_func = dict()
        self.num_actions = num_actions

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = str(x.astype(np.int32)) #no more depreciation errors now :D
        if x not in self.q_func:
            self.q_func[x] = np.zeros(self.num_actions, dtype=np.float32)
        
        for_df.loc[len(for_df.index)] = [self.q_func[x]]
        
        return self.q_func[x]

    def forward_batch(self, x_batch):
        return np.asarray([self.forward(x) for x in x_batch])

    def update_batch(self, s_batch, a_batch, delta_batch):
        for s, a, delta in zip(s_batch, a_batch, delta_batch):
            q_vals = self.forward(s)
            q_vals[a] += delta

    def update(self, s, a, delta):
        q_vals = self.forward(s)
        q_vals[a] += delta

    def get_action(self, x):
        return int(self.forward(x).argmax())

    def display(self):
        pprint(self.q_func)


class TabularQLearningAgent:
    """A Tabular. epsilon greedy Q-Learning Agent using Experience Replay """

    def __init__(self,
                 env,
                 seed=None,
                 lr=0.002,
                 training_steps=2500,
                 final_epsilon=0.1,
                 exploration_steps=2500,
                 gamma=0.99,
                 verbose=True,
                 **kwargs):

        # This implementation only works for flat actions
        assert env.flat_actions
        self.verbose = verbose
        if self.verbose:
            print("\nRunning Tabular Q-Learning with config:")
            pprint(locals())

        # set seeds
        self.seed = seed
        if self.seed is not None:
            np.random.seed(self.seed)

        # envirnment setup
        self.env = env

        self.num_actions = self.env.action_space.n
        self.obs_dim = self.env.observation_space.shape

        # logger setup
        self.logger = SummaryWriter()

        # Training related attributes
        self.lr = lr
        self.exploration_steps = 2500
        self.final_epsilon = final_epsilon
        self.epsilon_schedule = np.linspace(
            1.0, self.final_epsilon, self.exploration_steps
        )
        self.discount = gamma
        self.training_steps = training_steps
        self.steps_done = 0 #variable to be searched for steps_done

        # Q-Function
        self.qfunc = TabularQFunction(self.num_actions)

    def get_epsilon(self):
        if self.steps_done < self.exploration_steps:
            #added the df1.loc before the return statements
            df3.loc[len(df3.index)] = [self.epsilon_schedule[self.steps_done]]
            return self.epsilon_schedule[self.steps_done]

        df3.loc[len(df3.index)] = [self.final_epsilon]
        return self.final_epsilon

    def get_egreedy_action(self, o, epsilon):
        """
        There is no reason to get this as it is already captured by the 
        action_num_val
        """
        if random.random() > epsilon:
            return self.qfunc.get_action(o)
        return random.randint(0, self.num_actions-1)


    def optimize(self, s, a, next_s, r, done):
        # get q_val for state and action performed in that state
        q_vals_raw = self.qfunc.forward(s)
        q_val = q_vals_raw[a]

        # get target q val = max val of next state
        target_q_val = self.qfunc.forward(next_s).max()
        target = r + self.discount * (1-done) * target_q_val

        # calculate error and update
        td_error = target - q_val
        td_delta = self.lr * td_error

        # optimize the model
        self.qfunc.update(s, a, td_delta)

        s_value = q_vals_raw.max()
        df2.loc[len(df1.index)] = [td_error, s_value, target_q_val, td_delta]
        return td_error, s_value
    def train(self): # used in the main function
        if self.verbose:
            print("_______")
        
        step_num = []

        num_episodes = 0
        training_steps_remaining = self.training_steps

        while self.steps_done < self.training_steps:
            ep_results = self.run_train_episode(training_steps_remaining)
            ep_return, ep_steps, goal = ep_results
            num_episodes += 1
            training_steps_remaining -= ep_steps

            self.logger.add_scalar("episode", num_episodes, self.steps_done)
            self.logger.add_scalar(
                "epsilon", self.get_epsilon(), self.steps_done
            )
            self.logger.add_scalar(
                "episode_return", ep_return, self.steps_done
            )
            self.logger.add_scalar(
                "episode_steps", ep_steps, self.steps_done
            )
            self.logger.add_scalar(
                "episode_goal_reached", int(goal), self.steps_done
            )
            
            if num_episodes % 10 == 0 and self.verbose:
                print(num_episodes)
                print(f"\nEpisode {num_episodes}:")
                print(f"\tsteps done = {self.steps_done} / "
                      f"{self.training_steps}")
                print(f"\treturn = {ep_return}")
                print(f"\tgoal = {goal}")
                step_num.append(self.steps_done)
            
            self.step_num = step_num #now a part of the object
            self.eps_num = num_episodes

                

            temp = self.steps_done / self.training_steps
            df1.loc[len(df1.index)] = [num_episodes,temp, ep_return, goal, num_episodes]

        self.logger.close()
        if self.verbose:
            print("Training complete")
            print(f"\nEpisode {num_episodes}:")
            print(f"\tsteps done = {self.steps_done} / {self.training_steps}")
            print(f"\treturn = {ep_return}")
            print(f"\tgoal = {goal}")

    def get_step(self):
        return self.step_num, self.eps_num

    def get_a_space(self):
        return self.num_actions,self.obs_dim
    def run_train_episode(self, step_limit): 
        s = self.env.reset()
        done = False
        env_limit_reached = False

        steps = 0
        episode_return = 0

        while not done and not env_limit_reached and steps < step_limit:
            a = self.get_egreedy_action(s, self.get_epsilon())
            self.action_num_val = a

            try:
                """
                Abhay suggested the code used by a_verbose in order to get the action taken
                """
                a_verbose = self.env.action_space.get_action(a) 
            except:
                a_verbose = -1
            next_s, r, done, env_limit_reached = self.env.step(a)

            self.steps_done += 1
            td_error, s_value = self.optimize(s, a, next_s, r, done)
            self.logger.add_scalar("td_error", td_error, self.steps_done)
            self.logger.add_scalar("s_value", s_value, self.steps_done)

            s = next_s
            episode_return += r
            steps += 1
            df4.loc[len(df4.index)] = [a, a_verbose, td_error, s_value]

        return episode_return, steps, self.env.goal_reached() 

    def run_eval_episode(self, #also used in the main function
                         env=None,
                         render=False,
                         eval_epsilon=0.05,
                         render_mode="readable"):
        if env is None:
            env = self.env
        s = env.reset()
        done = False

        steps = 0
        episode_return = 0
        
        """
        decided to just use a temp variable instead of using an 
        array and the .sum() method to hold sums from the equation
        """

        value = 0
        temp_v = 0
        new_v = 0
        

        line_break = "="*60
        if render:
            print("\n" + line_break)
            print(f"Running EVALUATION using epsilon = {eval_epsilon:.4f}")
            print(line_break)
            env.render(render_mode)
            input("Initial state. Press enter to continue..")

        while not done:
            a = self.get_egreedy_action(s, eval_epsilon)
            next_s, r, done, Actions = env.step(a)
            s = next_s
            episode_return += r
            steps += 1
            a_verbose = self.env.action_space.get_action(a)
            


            """
            doing this as changing the code near get_action may cause
            many issues. May be good for in the future to write some code
            wherever this file is to get a way to get the probability much like
            the 'action_space.get_action(a)' 
            """
            text = str(a_verbose)
            result = text.index("prob=")
            probability = text[result+5:result+9]
            probability = float(probability)

            result = text.index("target=")
            target_host = text[result+8:result+12]
            

            #Have recently learned the above code does not work as intended. 
            #So I am currently reading the code to do someting similar to the get_step() function as I did earleir

            #value function
            gamma = self.discount
            action = int(a)
            reward = int(episode_return)
            prob = probability
            
            


            df5.loc[len(df5.index)] = [gamma, reward, prob, target_host]


            if render:
                print("\n" + line_break)
                print(f"Step {steps}")
                print(line_break)
                print(f"Action Performed = {env.action_space.get_action(a)}")
                env.render(render_mode)
                print(f"Reward = {r}")
                print(f"Done = {done}")
                input("Press enter to continue..")

                if done:
                    print("\n" + line_break)
                    print(line_break)
                    print(f"Goal reached = {env.goal_reached()}")
                    print(f"Total steps = {steps}")
                    print(f"Total reward = {episode_return}")
    

        
            
            
        return episode_return, steps, env.goal_reached() #The data I use of the second DataFrame



#decided to create a class so to keep lots of the clutter outside the main function

class Auto_Analyze: 
    def __init__(self, step_num, eps_num):
        self.episode_seperator = {}
        self.step_num = step_num
        self.eps_num = eps_num


    def eps_seperation(self, step_num, eps_num):
        for num in step_num:
                temp_dict = {}
                var_name = "brk%d" % num
                temp_dict[var_name] = num
                print(temp_dict, num)
                self.episode_seperator.update(temp_dict)
        for key,value in self.episode_seperator.items():
            print(key, ":", value)

        return self.episode_seperator
    
    def __str__(self) -> str:
        return self.episode_seperator
    




if __name__ == "__main__":
    #6 dataframes to be combined later
    df1 = pd.DataFrame(columns=["num_eps", "temp", "ep_return", "goal", "episode_number"])
    df2 = pd.DataFrame(columns=["td_error", "s_value", "target_q_val", "td_delta"])
    df3 = pd.DataFrame(columns=["epsilon"])
    df4 = pd.DataFrame(columns=["action_num_val", "action_verbose", "td_error", "s_value"])
    df5 = pd.DataFrame(columns=["gamma", "reward", "probability", "target_host"])
    for_df = pd.DataFrame(columns=["q_func_value"])

    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("env_name", type=str, help="benchmark scenario name")
    parser.add_argument("--render_eval", action="store_true",
                        help="Renders final policy") #env? 
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate (default=0.001)")
    parser.add_argument("-t", "--training_steps", type=int, default=1000, #made this smaller so I could test and run faster
                        help="training steps (default=10000)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="(default=32)")
    parser.add_argument("--seed", type=int, default=0,
                        help="(default=0)")
    parser.add_argument("--replay_size", type=int, default=10000,
                        help="(default=100000)")
    parser.add_argument("--final_epsilon", type=float, default=0.05,
                        help="(default=0.05)")
    parser.add_argument("--init_epsilon", type=float, default=1.0,
                        help="(default=1.0)")
    parser.add_argument("-e", "--exploration_steps", type=int, default=10000,
                        help="(default=10000)")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="(default=0.99)")
    parser.add_argument("--quite", action="store_false",
                        help="Run in Quite mode")

    args = parser.parse_args()

    env = nasim.make_benchmark(
        args.env_name,
        args.seed,
        fully_obs=True,
        flat_actions=True,
        flat_obs=True
    )
    ql_agent = TabularQLearningAgent(
        env, verbose=args.quite, **vars(args)
    )



    
    rprint("[red]Starting...[/red]")
    ql_agent.train()
    ql_agent.run_eval_episode(render=args.render_eval)


    #looking for host data
    step_num, eps_num = ql_agent.get_step()
    actions = ql_agent.get_a_space
    rprint(actions)
    

    """
    The below code combines all the dataframes into one single dataframe
    I then tried to go online to find an implementation of a state-action
    value function. 

    """

    df2['index'] = range(1,len(df2) + 1)
    df3['index'] = range(1,len(df3) + 1)
    df4['index'] = range(1,len(df4) + 1)
    df5['index'] = range(1,len(df5) + 1)
    for_df['index']= range(1,len(for_df) + 1)

    #combine using merge
    result = pd.merge(df3, df4, on="index")
    result = pd.merge(result,df5, on ="index")
    result = pd.merge(result,for_df, on ="index")
    
    #convert the combined DataFrame into a .csv file
    result.to_csv("QL_Output.csv")
    
    rprint("[green]QL_Output.csv[/green]")


    """
    Create a number of variables depending on 
    the number of episodes. TODO later
    """

    #WIP
    #eps_seperation( step_num, eps_num)
    
    most_common_action=(result["action_num_val"].value_counts().nlargest(5))
    least_common_action=(result["action_num_val"].value_counts().nsmallest(5))

    """
    Calls for an action num val csv sheet
    to be created in order to best compare this
    with the most_common_action number 

    I am currently figuring out a way to better output these
    statistics rather than just printing their results back
    into the console. 
    """

    
    #**UNCOMMENT BELOW IF**
    #not using tiny, small, medium environment


    """
    ANV_legend = result.drop_duplicates(subset=["action_num_val"])
    ANV_legend = ANV_legend[["index", "action_num_val", "action_verbose"]]
    ANV_legend.sort_values(by=["action_num_val"])
    ANV_legend.to_csv("ANV_legend.csv")

    """


    #will correlate this to the ANV_legend.csv file
    single_most = (result["action_num_val"].value_counts().nlargest(1))
    single_most = (single_most[0])

    

    br = "\n"

    print("The 5 least common action preformed by the agent was:{}{}".format(br,least_common_action))
    print("The 5 most common action preformed by the agent was:{}{}".format(br,most_common_action))
    

        


    
    


     

    

    
