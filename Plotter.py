import matplotlib
import matplotlib.pyplot as plt

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython: from IPython import display


def plot(values, average):
    plt.figure(2)
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(values)

    plt.plot(average)
    plt.pause(0.001)
    # print("Episode", len(values), "\n", period, "episode moving avg:", average[-1])
    if is_ipython: display.clear_output(wait=True)

