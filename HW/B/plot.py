import pickle
import matplotlib.pyplot as plt
import numpy as np

task = (1,"c")
rewards = pickle.load(open("tasks/"+str(task[0])+"/"+task[1]+"/rewards.p", "rb"))
plt.plot(rewards)
plt.waitforbuttonpress()