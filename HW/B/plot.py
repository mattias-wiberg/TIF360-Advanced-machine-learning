import pickle
import matplotlib.pyplot as plt
import numpy as np

def moving_average(x, w):
    avg = []
    for i in range(len(x)):
        avg.append(np.mean(x[max(0, int(i-w/2)):min(len(x)-1,int(i+w/2))]))
    return avg

task = (1,"a")
rewards = pickle.load(open("tasks/"+str(task[0])+"/"+task[1]+"/rewards.p", "rb"))
plt.plot(rewards)
average = moving_average(rewards, 100)
np.mean(average)
plt.plot(average)
plt.legend(["Rewards", "Moving average"])
plt.title("Rewards for task "+str(task[0])+" "+task[1])
plt.savefig("tasks/"+str(task[0])+"/"+task[1]+"/rewards.png")
plt.show()