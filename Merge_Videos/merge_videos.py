from tkinter.filedialog import askopenfilenames, asksaveasfilename
from natsort import natsorted
import os

def main():
    print('Select videos to merge:')

    in_paths = askopenfilenames()
    in_paths = list(in_paths)
    natsorted(in_paths)

    print('These videos will be merged according to the order below:')
    for i in in_paths:
        print(i)

    print('Save the merged video as:')
    out_path = asksaveasfilename(defaultextension='.mp4', initialdir=in_paths[0], initialfile='mergedVideo')

    in_list = ''
    for i in in_paths:
        in_list += 'file \''+i+'\'\n'
    with open('list.txt', 'w') as list_file:
        list_file.write(in_list)

    # See http://trac.ffmpeg.org/wiki/Concatenate
    print('Running ffmpeg\\bin\\ffmpeg.exe -y -f concat -safe 0 -i list.txt -c copy "'+out_path+'"')
    os.system('ffmpeg\\bin\\ffmpeg.exe -y -f concat -safe 0 -i list.txt -c copy "'+out_path+'"')

    os.remove('list.txt')

if __name__ == '__main__':
    main()
