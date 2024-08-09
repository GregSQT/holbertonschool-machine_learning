#!/usr/bin/env python3
"""
Function used to plot a stacked bar graph
"""


import numpy as np
import matplotlib.pyplot as plt


"""
Function without input
"""


def bars():
    np.random.seed(5)
    fruit = np.random.randint(0, 20, (4, 3))
    plt.figure(figsize=(6.4, 4.8))

    names = ['Farrah', 'Fred', 'Felicia']
    fruit_name = ['apples', 'bananas', 'oranges', 'peaches']
    colors = ['red', 'yellow', '#ff8000', '#ffe5b4']

    for i in range(len(fruit)):
        plt.bar(names, fruit[i], bottom=np.sum(fruit[:i], axis=0),
            color=colors[i], label=fruit_name[i], width=0.5)

    plt.title("Number of Fruit per Person")
    plt.ylabel("Quantity of Fruit")
    plt.yticks(np.arange(0, 90, 10))
    plt.legend()
    plt.show()
