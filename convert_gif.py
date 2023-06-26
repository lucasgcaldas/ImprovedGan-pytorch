
'''
convert img_to_gif script
Copyright (c) Xiangzi Dai, 2019
'''
import imageio
import os
import sys

def create_gif(or_path, source, name, duration):
    """
    Create a GIF from a sequence of images.

    Args:
        or_path (str): The original path where the images are located.
        source (list): List of image filenames to include in the GIF.
        name (str): The filename of the output GIF.
        duration (float): The duration (in seconds) for each frame of the GIF.
    """
    frames = []
    for img in source:
        frames.append(imageio.imread(os.path.join(or_path, img)))
    imageio.mimsave(name, frames, 'GIF', duration=duration)
    print("ok")

def get_list(file_path):
    """
    Get a list of files ordered by their creation time.

    Args:
        file_path (str): The directory path containing the files.

    Returns:
        list: List of file names ordered by creation time.
    """
    dir_list = os.listdir(file_path)
    if not dir_list:
        return
    else:
        dir_list = sorted(dir_list, key=lambda x: os.path.getmtime(os.path.join(file_path, x)))
        return dir_list

def main(paths, start=None, end=None):
    """
    Main function to create a GIF from a directory of images.

    Args:
        paths (str): The target directory path containing the images.
        start (int, optional): The index of the first image to include in the GIF. Defaults to None.
        end (int, optional): The index of the last image to include in the GIF. Defaults to None.
    """
    pic_list = get_list(paths)
    gif_name = "result.gif"
    duration_time = 0.2  # duration time

    # Create GIF
    create_gif(paths, pic_list[start:end], gif_name, duration_time)

if __name__ == '__main__':
    """
    Usage:
        python3 convert_gif.py ./data_dir
        python3 convert_gif.py ./data_dir 80 600
    """
    nums = len(sys.argv)
    if nums == 2:
        main(sys.argv[1])
    elif nums == 3:
        main(sys.argv[1], int(sys.argv[2]))
    elif nums == 4:
        if int(sys.argv[2]) <= int(sys.argv[3]):
            main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
        else:
            print("start_num should be less than or equal to end_num")
    else:
        print("Please input valid parameters")
