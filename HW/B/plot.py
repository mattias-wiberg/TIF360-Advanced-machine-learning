import pickle
import matplotlib.pyplot as plt
import numpy as np

rewards = pickle.load(open("tasks/1/c/rewards.p", "rb"))
plt.plot(rewards)
plt.waitforbuttonpress()