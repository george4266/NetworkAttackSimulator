import yaml

"""
with how small it was, found no reason for this to
be a class. I don't think this file usage will be scaled
up to the point of needing a class.
"""

def read_file(filename):
        
        safe=0

        with open(filename, "r") as file:
            safe = yaml.safe_load(file)

        return safe

def dict_MDP(safe):

    """
        This function will get the specirfic parts for what is 
        needed for the MDP of the chosen yaml file

        - firewall -> P
        - subnet and topology -> S 
        - explits and privilege escalation -> A

        Are the main parts of the yaml needed to then make changes
        to the MDP/ network model as needed. 
    """

    firewall = safe.get('firewall') #P

    subnets = safe["subnets"] #S
    topology = safe["topology"]#S
    sensitive = safe["sensitive_hosts"]#S

    exploits = safe["exploits"]#A
    escalation = safe["privilege_escalation"]#A

    P, S, A = firewall, [subnets, topology, sensitive], [exploits, escalation]

    #Probability, States, Action

    return P, S, A 
    

my_file = read_file("small.yaml")
prob, state, actions = dict_MDP(my_file)

#output the part related to firewall.
print("Firewall: {}".format(prob))




