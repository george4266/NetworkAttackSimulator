import plotly.express as px
import pandas as pd

df = pd.read_csv("saved_out/bruteforce_out.csv", usecols=["Topology"])
print("success!")
print(df)

