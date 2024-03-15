import cv2
import os
import re
import glob


def sort_key(filename):
    # This function extracts the numerical part of the filename and returns it as an integer
    numbers = re.findall(r'\d+', filename)
    return int(numbers[0]) if numbers else None


# Directory containing images
img_dir = 'framePlotsNormal'  # Enter Directory of all images
data_path = os.path.join(img_dir, '*.png')
files = sorted(glob.glob(data_path), key=sort_key)

# Frame properties
frame = cv2.imread(files[0])
height, width, layers = frame.shape

# Video properties
video_name = 'output_video_normal_33.avi'
fps = 3  # Frames per second
fourcc = cv2.VideoWriter_fourcc(*'DIVX')

# Video writer object
video = cv2.VideoWriter(video_name, fourcc, fps, (width, height))

for f in files:
    img = cv2.imread(f)
    video.write(img)

cv2.destroyAllWindows()
video.release()
