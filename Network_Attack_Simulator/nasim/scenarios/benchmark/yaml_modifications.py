import re
import yaml

class YAML_mod:

    def __init__(self, filename):
        self.filename = filename

        if isinstance(filename, list):
            param_type = 1
        if isinstance(filename, str):
            param_type = 2
        
        self.param_type = param_type


    def read_file(self):
        filename = self.filename
        param_type = self.param_type
        safe = ""

        """   if param_type == 1:
            all_files = {filename: open(filename, "r") for filename in self.filename}
            safe = str(all_files)
        """

        if param_type == 2:
            with open(filename, 'r') as file:
                safe = yaml.safe_load(file)
        
        self.safe = safe
        return safe

    def get_parts(self):
        """
            This function will get the specirfic parts for what is 
            needed for the MDP of the chosen yaml file

            - firewall -> P
            - subnet and topology -> S 
            - explits and privilege escalation -> A

            Are the main parts of the yaml needed to then make changes
            to the MDP/ network model as needed. 
        """
        safe = self.safe

        firewall = safe["firewall"]

        subnets = safe["subnets"]
        topology = safe["topology"]
        sensitive = safe["sensitive_hosts"]

        exploits = safe["exploits"]
        escalation = safe["privilege_escalation"]


        self.states = firewall 

        self.probability = subnets, topology, sensitive, sensitive 

        self.actions = exploits,escalation

        return self.states, self.probability, self.actions






    def __str__(self):
        return self.filename

    
yaml_list = "small.yaml"


read_yaml = YAML_mod(yaml_list)
read_yaml = read_yaml.read_file()
read_yaml = read_yaml.get_parts()



