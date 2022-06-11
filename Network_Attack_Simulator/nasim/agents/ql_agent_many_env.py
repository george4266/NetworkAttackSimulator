from ql_agent import * 

"""
Using the QL_agent class try to create a new file
that allows to save data of multiple runs of the same environment
"""

df_ql = pd.DataFrame(columns=["reward", "diff_in_reward", "steps", "actions"])