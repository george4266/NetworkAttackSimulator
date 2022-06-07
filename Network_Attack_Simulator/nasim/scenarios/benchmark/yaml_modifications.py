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

        if param_type == 1:
            pass

        if param_type == 2:
            with open(filename, 'r') as file:
                safe = yaml.safe_load(file)
        
        return safe


    def __str__(self):
        return self.filename

    
yaml_list = ["tiny.yaml", "small.yaml"]


read_yaml = YAML_mod(yaml_list)
print(read_yaml.what_type())
read_yaml = read_yaml.read_file()
print(read_yaml)



