import yaml

def read_full(filename):
    with open(filename, "r") as file:
        safe = yaml.safe_load(file)

    return safe

print(read_full("tiny.yaml"))


def read_part(part, filename):
    with open(filename, "r") as file:
        safe = yaml.safe_load(file)
    return safe[part]


print(read_part("subnets","tiny.yaml"))

print(read_part("firewall", "tiny.yaml"))