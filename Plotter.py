import matplotlib
import matplotlib.pyplot as plt

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython: from IPython import display


def plot(values, period):
    plt.figure(2)
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(values)

    average = get_average(values, values)
    plt.plot(average)
    plt.pause(0.001)
    print("Episode", len(values), "\n", period, "episode moving avg:", average[-1])
    if is_ipython: display.clear_output(wait=True)


def get_average(values, period):
    if len(values) >= period:
        average = sum(values[len(values) - period:]) / period
        return average
    else:
        return 0
