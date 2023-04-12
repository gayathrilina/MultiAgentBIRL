import matplotlib.pyplot as plt
import numpy as np
import pickle

# def sampleNewWeight(dims, options, seed=None):
#     np.random.seed(seed)
#     # np.random.seed(None)
#     lb = options.lb 
#     ub = options.ub    
#     if options.priorType == 'Gaussian':
#         # w0 = options.mu + np.random.randn(dims, 1)*options.sigmasq  # Direct way to do it
#         # for i in range(len(w0)):
#         #     w0[i] = max(lb, min(ub, w0[i])) # Check to ensure weights are within bounds

#         mean = np.ones(dims) * options.mu
#         cov = np.eye(dims) * options.sigmasq
#         w0 = np.clip(np.random.multivariate_normal(mean, cov), a_min=lb, a_max=ub).reshape((dims, 1))
#     else:
#         w0 = np.random.uniform(low=lb, high=ub, size=(dims,1))
#     return w0


# import os
# import numpy as np
# import matplotlib.pyplot as plt
# import imageio

# # Build GIF
# with imageio.get_writer('mygif.gif', mode='I') as writer:
#     for filename in ['rewards1.png', 'rewards2.png', 'rewards3.png', 'rewards4.png']:
#         image = imageio.imread(filename)
#         writer.append_data(image)


# import numpy as np
# from matplotlib import pyplot as plt
# from matplotlib import animation

# # Setting up the figure, the axis, and the plot element we want to animate
# fig = plt.figure()
# ax = plt.axes(xlim=(0, 5), ylim=(0, 6))
# line, = ax.plot([], [], lw=2)

# # initialization method: plot the background of each frame
# def init():
#     line.set_data([], [])
#     return line,

# # animation method.  This method will be called sequentially
# pow_ = 0
# def animate(i):
#     global pow_
#     x = [1, 2, 3, 4]
#     y = r[pow_]
#     pow_+=1
#     line.set_data(x, y)
#     return line,

# # call the animator.  blit=True means only re-draw the parts that have changed.
# anim = animation.FuncAnimation(fig, animate, init_func=init,
#                                frames=3, interval=400, blit=True)

# ########

# import matplotlib.pyplot as plt
# import numpy

# hl, = plt.plot([], [])

# def update_line(hl, new_data):
#     hl.set_xdata(numpy.append(hl.get_xdata(), new_data))
#     hl.set_ydata(numpy.append(hl.get_ydata(), new_data))
#     plt.draw()

# for i in range(len(r)):
#     update_line(hl, r[i])

# #########

# import matplotlib.pyplot as plt
# # generate axes object
# ax = plt.axes()

# # set limits
# plt.xlim(0,10) 
# plt.ylim(0,10)
# line_1  = plt.plot([1, 1, 3, 1], label='original', color = "green")

# for i in range(len(r)): 
#     line_2  = plt.plot(r[i], label='learned', color = "red")
#     plt.legend()
#     plt.pause(0.5)
#     line = line_2.pop(0)
#     line.remove()


    
    
#     #line_2.remove()

# ax = plt.figure().gca()
# #plt.gcf().set_size_inches(10, 10)
# plt.xlim(0, 3)
# plt.ylim(0, 2)
# plt.plot(obs[0][0] + 0.5, obs[0][1] + 0.5, marker='o', markersize=20, color="red", label = "agent 1")
# plt.plot(obs[1][0] + 0.5, obs[1][1] + 0.5, marker='o', markersize=20, color="blue", label = "agent 2")
# plt.plot(obs[2][0] + 0.5, obs[2][1] + 0.5, marker='s', markersize=20, color="green", label = "food 1")
# plt.plot(obs[3][0] + 0.5, obs[3][1] + 0.5, marker='s', markersize=20, color="orange", label = "food 2")
# ax.set_xticks(range(0, 3))
# ax.set_yticks(range(0, 2))


# plt.grid(True)
# plt.show()

file = "./results/3.2grid-2agent2food-uniformprior-23.45"
num_row = 3
num_col = 2
step = 1

with open(file + '/results.pickle', 'rb') as f:
    x = pickle.load(f)

for _ in x['expert_trajectory']:
    obs = _[0]
    print(obs)
    ax = plt.figure().gca()
    #plt.gcf().set_size_inches(10, 10)
    plt.xlim(0, num_row)
    plt.ylim(0, num_col)
    plt.plot(obs[0][0] + 0.5, obs[0][1] + 0.5, marker='o', markersize=20, color="red", label = "agent 1")
    plt.plot(obs[1][0] + 0.5, obs[1][1] + 0.5, marker='o', markersize=20, color="blue", label = "agent 2")
    plt.plot(obs[2][0] + 0.5, obs[2][1] + 0.5, marker='s', markersize=20, color="green", label = "food 1")
    plt.plot(obs[3][0] + 0.5, obs[3][1] + 0.5, marker='s', markersize=20, color="orange", label = "food 2")
    ax.set_xticks(range(0, 3))
    ax.set_yticks(range(0, 2))
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    plt.grid(True)
    plt.title("Experts | Step " + str(step+1))
    plt.savefig(file + "/Exp-Step " + str(step) + ".png")
    #plt.show()
    plt.close
    if step == len(x['expert_trajectory']):
        ax = plt.figure().gca()
        #plt.gcf().set_size_inches(10, 10)
        plt.xlim(0, num_row)
        plt.ylim(0, num_col)
        plt.plot(obs[0][0] + 0.5, obs[0][1] + 0.5, marker='o', markersize=20, color="red", label = "agent 1")
        plt.plot(obs[1][0] + 0.5, obs[1][1] + 0.5, marker='o', markersize=20, color="blue", label = "agent 2")
        ax.set_xticks(range(0, 3))
        ax.set_yticks(range(0, 2))
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        plt.grid(True)
        plt.title("Experts | Step " + str(step+1))
        plt.savefig(file + "/Exp-Step " + str(step+1) + ".png")
        #plt.show()
        plt.close
    step += 1

step = 1
for _ in x['learner_trajectory']:
    obs = _[0]
    print(obs)
    ax = plt.figure().gca()
    #plt.gcf().set_size_inches(10, 10)
    plt.xlim(0, num_row)
    plt.ylim(0, num_col)
    plt.plot(obs[0][0] + 0.5, obs[0][1] + 0.5, marker='o', markersize=20, color="red", label = "agent 1")
    plt.plot(obs[1][0] + 0.5, obs[1][1] + 0.5, marker='o', markersize=20, color="blue", label = "agent 2")
    plt.plot(obs[2][0] + 0.5, obs[2][1] + 0.5, marker='s', markersize=20, color="green", label = "food 1")
    plt.plot(obs[3][0] + 0.5, obs[3][1] + 0.5, marker='s', markersize=20, color="orange", label = "food 2")
    ax.set_xticks(range(0, 3))
    ax.set_yticks(range(0, 2))
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    plt.grid(True)
    plt.title("Learners | Step " + str(step))
    plt.savefig(file + "/Learn-Step " + str(step) + ".png")
    #plt.show()
    plt.close
    if step == len(x['learner_trajectory']):
        ax = plt.figure().gca()
        #plt.gcf().set_size_inches(10, 10)
        plt.xlim(0, num_row)
        plt.ylim(0, num_col)
        plt.plot(obs[0][0] + 0.5, obs[0][1] + 0.5, marker='o', markersize=20, color="red", label = "agent 1")
        plt.plot(obs[1][0] + 0.5, obs[1][1] + 0.5, marker='o', markersize=20, color="blue", label = "agent 2")
        ax.set_xticks(range(0, 3))
        ax.set_yticks(range(0, 2))
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        plt.grid(True)
        plt.title("Learners | Step " + str(step+1))
        plt.savefig(file + "/Learn-Step " + str(step+1) + ".png")
        #plt.show()
        plt.close
    step += 1
