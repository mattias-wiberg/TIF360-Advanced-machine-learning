import pickle
import matplotlib.pyplot as plt
import numpy as np

rewards = pickle.load(open("rewards_task1a.p", "rb"))
plt.plot(rewards)
plt.waitforbuttonpress()