import pandas as pd
import numpy as np
import sys
print("0")
dat = pd.read_csv("D:/LAN-shared Hippocampus/1053/OF/1053_OF_source_extraction/frames_1_28081/LOGS_25-Mar_12_06_41/C_raw_26_Mar.csv", header = None)
print("1")
def neuronkeep(neuron1, neuron2, dat = dat):
    df = dat.T
    df.columns = list(range(1, len(dat.index)+1))
    corr = df.corr()

    pairs = []
    print("2")
    for i in range(1, len(corr.columns)+1):
        boolean = list(corr[i].apply(lambda x: True if x >= 0.7 else False))
        for j in range(len(boolean)):
            if boolean[j] == True and corr[i][j+1] != 1.0:
                pairs.append([i, j+1, corr[i][j+1]])

    print("3")
    data = df * 25.7098 #C_raw * max(neuron.A(:, 1))
    for pair in pairs:
        if (pair[0] == neuron1 or pair[1] == neuron1) and (pair[0] == neuron2 or pair[1] == neuron2):
            avg1 = np.mean(data[neuron1])
            var1 = np.sum(((data[neuron1])-avg1)**2)
            avg2 = np.mean(data[neuron2])
            var2 = np.sum(((data[neuron2])-avg2)**2)
            keep = None
            if var1 > var2:
                keep = neuron1
            else:
                keep = neuron2
            print("Suggested neuron kept: " + str(keep))
            break
        else:
            if pair == pairs[-1]:
                print("Sorry, they are not highly correlated")
            else:
                continue

neuron1 = int(sys.argv[1])
neuron2 = int(sys.argv[2])
sys.stdout.write(str(neuronkeep(neuron1,neuron2,dat=dat)))
