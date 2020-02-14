import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("matches.csv")

print(dataset.head())

plt.style.use('ggplot')
plt.hist(dataset["season"], rwidth=0.9)
plt.show()
