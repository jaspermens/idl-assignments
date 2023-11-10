import numpy as np
import pickle as pk
import matplotlib.pyplot as plt
import os
from pathlib import Path
import matplotlib as mpl
# from clock_classification import *

def read_data():
    with open('plotdata.pk', 'rb') as f:
        data = pk.load(f)
    print(data.keys())
    return data

def plot_data():
    data = read_data()

    colors = [mpl.colors.hsv_to_rgb([0.408675, 0.36138614, r]) for r in np.linspace(.1, .9, len(data['filenames']), endpoint=True)]
#   colors  = plt.cm.viridis(np.linspace(0,1, len(classifiers_sorted)))
    colors[0] = 'darkorange'
#     colors = [[129/255, 202/255, 162/255], [0,0,0]] # first color is black, last is red
    # cm = mpl.colors.LinearSegmentedColormap.from_list(
                            # "Custom", colorbounds, N=20)
    
    # colors = cm(np.linspace(0,1,len(classifiers_sorted)))
    fig, (ax1, ax2) = plt.subplots(1,2, 
                                   figsize=[7,2.65], 
                                   dpi=200, 
                                #    layout='tight', 
                                   sharey=True, 
                                   sharex=True,
                                   )
     
    plt.subplots_adjust(
        top=0.895,
        bottom=0.17,
        left=0.075,
        right=0.98,
        hspace=0.1,
        wspace=0.025
    )

    ax1.grid(True, axis='y')
    ax2.grid(True, axis='y')
    
    ax1.set_ylim(0.46, 1)
    ax1.set_xlim(2, 145)
    
    ax1.set_xlabel('Epoch')
    ax2.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')

    ax1.set_title('Test')
    ax2.set_title('Train')

    for i, color in enumerate(colors):
        ax2.plot(data['train_history'][i], c=color)
        ax1.plot(data['val_history'][i], c=color)

        ax1.scatter(ax1.get_xlim()[1]-1, 
                    data['final_accuracies'][i], 
                    color=color, 
                    marker='<', 
                    zorder=2, 
                    s=20)
        
        # ax1.axhline(data['final_accuracies'][i],
        #           ls='dashed',
        #           alpha=.8,
        #           lw=1.2,
        #           c=color,
        #           zorder=0,
        #           )

    plt.savefig('reportfigs/accuracy_curves.pdf')
    plt.savefig('reportfigs/accuracy_curves.png')
    plt.show()


    
if __name__ in '__main__':
    plot_data()