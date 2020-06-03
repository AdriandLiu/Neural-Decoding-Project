import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, TensorDataset
from sklearn.model_selection import train_test_split

def distOneF(data,referencePts):
    '''
    distance calculation for one free-moving mouse data

    referencePts: two elements array/list/tuple
        bullying mouse position in pixels
    '''
    fixedX = referencePts[0]
    fixedY = referencePts[1]
    x = data.x
    y = data.y
    dist = np.sqrt((x - fixedX)**2 + (y - fixedY)**2)
    return dist


def distTwoF(data, pos):
    '''
    data: dlc data
    pos: "head", "body", "tail" for distance calculation
    '''
    if pos == "head":
        return np.sqrt((data["x"] - data["x.3"])**2 + (data["y"] - data["y.3"])**2)
    elif pos == "body":
        return np.sqrt((data["x.1"] - data["x.4"])**2 + (data["y.1"] - data["y.4"])**2)
    else:
        return np.sqrt((data["x.2"] - data["x.5"])**2 + (data["y.2"] - data["y.5"])**2)



def fourPointTransform(pts, dwd, dht, image = None):
    '''
    Task: Transform image by its corner four points.

    PARAMETERS:
    -----------
    image: array, Optional if intend to transform an image
        image matrix

    pts: list, tuple, array of list
        coordinates of the top-left, top-right, bottom-right, and bottom-left points

    dwd: width of the destination image

    dht: height of the destination image
    '''
    dst = np.array([[0, 0], [dwd-1, 0], [dwd-1, dht-1], [0, dht-1]], dtype="float32")

    # Transformation matrix
    tmat = cv2.getPerspectiveTransform(pts, dst)

    if image is None:
        return tmat
    else:
        # Apply the matrix
        warped = cv2.warpPerspective(image, tmat, (dwd, dht))
        return warped, tmat


def locCoordConvert(data, pts, dwd, dht):
    '''
    Task: Convert mouse location data to prospective coordicates after correcting coord system.

    PARAMETERS:
    -----------
    data: DataFrame
        original location data

    pts: list, tuple, array of list
        coordinates of the top-left, top-right, bottom-right, and bottom-left points

    dwd: width of the destination image

    dht: height of the destination image
    '''
    # Transposed transformation matrix
    tp_tmat = fourPointTransform(pts, dwd, dht)
    columns = [i for i in data.columns if i[0] == "x" or i[0] == "y"]
    transformed = pd.DataFrame()
    # x, y, 1
    for i in range(0,len(columns),2):
        temp = pd.concat([data[[columns[i],columns[i+1]]],
                        pd.DataFrame([1]*len(data))], axis = 1, join = "inner").values
        transform = pd.DataFrame(np.dot(temp, tp_tmat)[:,:2])

        transformed = pd.concat([transformed, transform], ignore_index = True, axis=1)
    transformed.columns = columns
    # Since (transformed head)^T = (transformation matrix)(head)^T, and (AB)^T = B^TA^T
    # Transformed head = head(transformation matrix)^T

    return transformed

def ptsCoordConvert(refPts, pts, dwd, dht):
    '''
    Task: Convert user-specifed points to prospective coordicates after correcting coord system.

    PARAMETERS:
    -----------
    refPts: list, tuple, array of list
        coordinates of the top-left, top-right, bottom-right, and bottom-left points

    pts: list, tuple, array of list
        coordinates of points to convert

    dwd: width of the destination image

    dht: height of the destination image
    '''
    # Transformation matrix
    tmat = fourPointTransform(refPts, dwd, dht)
    transformedPts = []
    # Converse mutiple points
    if np.array(pts).shape != (2,):
        for i in pts:
            i.append(1)
            transformedPts.append(list(np.dot(tmat, i)[:2]))
    # Single point
    else:
        pts.append(1)
        transformedPts.append(list(np.dot(tmat, pts)[:2]))

    return transformedPts
def align(neuron_data, dlc_data, timestamp, gap_time):
    '''
    Task: align neuron data and dlc data based on the corresponding timestamp.dat. The alignment is followed by frame number

    PARAMETERS:
    -----------
    neuron_data: cnmfe data, transposed

    dlc_data: deeplabcut data

    timestamp: timestamp file in the specific mouse folder

    return: sorted msCam, sorted behavCam
    '''
    new_order = []
    # Check the diff between cam and behav in timestamp.
    camNum = list(set(timestamp.camNum))
    redundant = list(set(timestamp[timestamp["camNum"]==camNum[0]]["frameNum"]) - set(timestamp[timestamp["camNum"]==camNum[1]]["frameNum"])) # may cause NAN value afterwards, so remove it now
    for i, index in zip(timestamp["frameNum"].values, timestamp.index):
        if i not in redundant:
            continue
        else:
            timestamp = timestamp.drop(index)

    # We do not need coords column
    dlc_data = dlc_data.drop(columns = "coords", axis = 1)
    # For length of dlc and neuron data is not the same, take out the redundant data (may be caused by lack of data while integrating behavioral video)

    min_len = min(len(neuron_data), len(dlc_data), len(timestamp), len(timestamp[timestamp["camNum"]==camNum[0]]), len(timestamp[timestamp["camNum"]==camNum[0]]))
    neuron_data = neuron_data.iloc[0:min_len:]
    dlc_data = dlc_data.iloc[0:min_len:]
    timestamp["frameNum"] = timestamp["frameNum"] - gap_time + 1 #change of index
    timestamp = timestamp[timestamp["frameNum"]<=min_len]
    timestamp.index = range(0,len(timestamp))

    try:
        for ms, behav in zip(timestamp["camNum"],timestamp["frameNum"]):
            if ms == 0:
                new_order.append(neuron_data.iloc[behav-1].values) # -1 becuase frameNum start from 1 while neuron_data start from 0
            else:
                new_order.append(dlc_data.iloc[behav-1].values)
    except IndexError:
        print("Neuron data and dlc data are not in the same length, fix by checking the video length for each")
    merge_data = pd.concat([pd.DataFrame(new_order), timestamp[["camNum","frameNum"]]], axis = 1).sort_values(by = "frameNum")
    msCam = merge_data[merge_data["camNum"]==camNum[0]].dropna(axis = 1).drop(columns = ["camNum","frameNum"], axis = 1)
    msCam.index = range(1,len(msCam)+1) # for later concatenate
    msCam.columns = range(0,len(msCam.columns))

    behavCam = merge_data[merge_data["camNum"]==camNum[1]].dropna(axis = 1).drop(columns = ["camNum","frameNum"], axis = 1)
    behavCam.columns = dlc_data.columns
    behavCam.index = range(0,len(behavCam)) # for later concatenate
    return (msCam,behavCam)

def dataPrep(filename, split_frac, scenario, corner_pts, cage_dim, refer_pt, dist_thres, gap_time, batch_size):
    '''
    Task: Prepare data for feeding DL model

    PARAMETERS:
    ------------

    filename: dict
        Key: neuron_A, neuron_B, dlc_A, dlc_B, timestamp_A, timestamp_B
        Value: their corresponding file path and names
    split_frac: decimal
        split frac for train_val_test split
    scenario: str
        "one": one free-moving mouse or "two": two free-moving mice
    corner_pts: numpy array with data type np.float32
        cage four corner coordinate points, upper left, upper right, lower right, lower left
    cage_dim: int
        cage dimension in centimeters
    refer_pt: tuple, list
        bullying mouse position in pixels
    dist_thres: int
        the distance threshold between two mice, <threshold = interacted, >threshold = not interacted
    gap_time: int
        time of no mouse shows up in the cage, in frames
    batch_size: int
        model batch size
    '''

    gap_time_A = gap_time
    gap_time_B = gap_time

    dlc_A = pd.read_csv(filename['dlc_A'], skiprows = 2).iloc[gap_time_A:,]
    dlc_B = pd.read_csv(filename['dlc_B'], skiprows = 2).iloc[gap_time_B:,]


    neuron_A = pd.read_csv(filename['neuron_A'], header = None).T
    neuron_B = pd.read_csv(filename['neuron_B'], header = None).T.iloc[gap_time_B:,]
    timestamp_A = pd.read_csv(filename['timestamp_A'], \
    sep='\t', header = None, skiprows=1, names = ["camNum","frameNum","sysClock","buffer"])
    timestamp_B = pd.read_csv(filename['timestamp_B'], \
    sep='\t', header = None, skiprows=1, names = ["camNum","frameNum","sysClock","buffer"])
    timestamp_A = timestamp_A[timestamp_A["frameNum"]>=gap_time_A]
    timestamp_B = timestamp_B[timestamp_B["frameNum"]>=gap_time_B]


    msCam, behavCam = align(neuron_A, dlc_A, timestamp_A, gap_time_A)      # alignment[0] == aligned neurons_1053B; alignment[1] == aligned dlc_1053B
    pts = corner_pts                                                       # four corner points
    newLoc = locCoordConvert(behavCam,pts,cage_dim[0],cage_dim[1])                            # convert to new location data with new dimension
    if scenari == "one":
        referPt = ptsCoordConvert(pts, refer_pt, cage_dim[0],cage_dim[1])[0]                    # convert bullying mouse location with new dimension
        dist = distOneF(newLoc, referPt)
    else:
        dist = distTwoF(newLoc, "head")                                  # calculate distance between bullying and defeated mouse
    labeled = [1 if i < dist_thres else 0 for i in dist]                            # if dist < 15, label 1 (has interaction), else 0 (no interaction)


    data = pd.concat([msCam, pd.DataFrame(labeled)], axis=1).dropna(axis = 0)
    data.columns = list(range(1,len(msCam.columns)+2))                      # avoid duplicate column name
    data = data.rename(columns={len(msCam.columns)+1:"interaction"})

    # One hot encoding
    one_hot = pd.get_dummies(data['interaction'])
    one_hot.columns = ["interaction.a", "interaction.b"]
    data = data.drop("interaction", axis = 1).join(one_hot)

    frac = split_frac
    x_train, x_test, y_train, y_test = \
            train_test_split(data[list(range(1,len(data.columns)-1))], data[["interaction.a", "interaction.b"]], test_size=frac, random_state=0)


    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    x_train = x_train.drop(1, axis = 1)
    y_embedding = np.zeros((len(x_train), len(x_train.columns)))

    x_train_tensor = torch.from_numpy(np.array(x_train)).float().to(device)
    y_train_tensor = torch.from_numpy(np.array(y_train)).long().to(device)
    train = TensorDataset(x_train_tensor, y_train_tensor)
    train_loader = torch.utils.data.DataLoader(dataset=train, batch_size=batch_size, shuffle=True)
    total_step = len(train_loader)


    x_val_tensor = torch.from_numpy(np.array(x_val)).float()
    y_val_tensor = torch.from_numpy(np.array(y_val)).float()
    val= TensorDataset(x_val_tensor, y_val_tensor)
    val_loader = torch.utils.data.DataLoader(dataset=test,batch_size=batch_size, shuffle=True)


    x_test_tensor = torch.from_numpy(np.array(x_test)).float()
    y_test_tensor = torch.from_numpy(np.array(y_test)).float()
    test = TensorDataset(x_test_tensor, y_test_tensor)
    test_loader = torch.utils.data.DataLoader(dataset=test,batch_size=batch_size, shuffle=True)

    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    filename = {'dlc_A':"//DMAS-WS2017-006/E/A RSync FungWongLabs/DLC_Data/1053 SI_A, Mar 22, 9 14 20/videos/\
1056 SI_A, Mar 22, 12 45 13DeepCut_resnet50_1053 SI_A, Mar 22, 9 14 20Jul31shuffle1_600000.h.csv",
'dlc_B':"//DMAS-WS2017-006/E/A RSync FungWongLabs/DLC_Data/1053 SI_A, Mar 22, 9 14 20/videos/\
1056 SI_B, Mar 22, 12 52 59DeepCut_resnet50_1053 SI_A, Mar 22, 9 14 20Jul31shuffle1_600000.h.csv",
'neuron_A':"//Dmas-ws2017-006/e/A RSync FungWongLabs/CNMF-E/1056/SI/1056_SI_A_Substack (240-9603)_source_extraction/frames_1_9364/LOGS_15-Sep_13_52_07/1056SI_A_240-9603.csv",
'neuron_B':"//Dmas-ws2017-006/e/A RSync FungWongLabs/CNMF-E/1056/SI/1056_SI_B_source_extraction/frames_1_27256/LOGS_19-Apr_00_38_59/1056SI_B.csv",
'timestamp_A':"//DMAS-WS2017-006/H/Donghan's Project Data Backup/Raw Data/Witnessing/female/Round 8/3_22_2019/H12_M45_S13/timestamp.dat",
'timestamp_B':"//DMAS-WS2017-006/H/Donghan's Project Data Backup/Raw Data/Witnessing/female/Round 8/3_22_2019/H12_M52_S59/timestamp.dat"}
    split_frac = 0.3
    scenario = 'one'
    corner_pts = np.array([(85,100),(85,450), (425,440), (420,105)], np.float32)
    cage_dim = [44,44]
    refer_pt = [400,270]
    dist_thres = 15
    gap_time = 270
    batch_size = 128

    dataPrep(filename, split_frac, scenario, corner_pts, cage_dim, refer_pt, dist_thres, gap_time, batch_size)
