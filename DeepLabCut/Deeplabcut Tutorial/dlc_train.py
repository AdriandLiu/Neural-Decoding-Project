import deeplabcut
import tensorflow as tf
import os
from tqdm import tqdm
from time import time
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
path = r"\\Dmas-ws2017-006\e\RSync FungWongLabs\DLC_Data"

def get_size(start_path):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # skip if it is symbolic link
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)

    return total_size


def train_dlc(path):
    os.chdir(path)
    folders = [name for name in os.listdir(path) if os.path.isdir(name) and name[0].isdigit()]
    num_folder = len(folders)
    for i in range(len(folders)):
        start = time()

        if os.path.exists(r"%s\%s\labeled-data"%(path, folders[i])) and os.path.exists(r"%s\%s\config.yaml"%(path, folders[i])) \
        and os.path.exists(r"%s\%s\dlc-models\iteration-0"%(path, folders[i])) == False and get_size(r"%s\%s\labeled-data"%(path, folders[i])) > 0:
            path_config_file = r"%s\%s\config.yaml" %(path, folders[i])
            deeplabcut.check_labels(path_config_file)
            deeplabcut.create_training_dataset(path_config_file)
            deeplabcut.train_network(path_config_file, shuffle = 1, autotune = True, displayiters = 5000, saveiters = 5000, maxiters = 200000)
            deeplabcut.analyze_videos(path_config_file, r"%s\%s\videos"%(path, folders[i]), videotype = '.mp4', save_as_csv = True)
            print("%s training has been done, %s left"%(folders[i], num_folder - i))
        elif os.path.exists(r"%s\%s\dlc-models\iteration-0"%(path, folders[i])):
            path_config_file = r"%s\%s\config.yaml" %(path, folders[i])
            print("%s model has been trained, do you want to retrain it? y/n"%(folders[i]))
            feedback = input()
            if feedback == "y":
                deeplabcut.train_network(path_config_file, shuffle = 1, autotune = True, displayiters = 5000, saveiters = 5000, maxiters = 200000)
            # If model was previously trained, read config.yaml to retrieve
            deeplabcut.analyze_videos(path_config_file, r"%s\%s\videos"%(path, folders[i]), videotype='.mp4', save_as_csv=True)
            print("%s training has been done, %s left"%(folders[i], num_folder - i))

        else:
            print("labeled-data folder not found OR empty OR config not found")
        print("Running time for %s is %s sec" %(folders[i], time() - start))


if __name__ == '__main__':
    train_dlc(path)

print("All DLC models have been trained")
