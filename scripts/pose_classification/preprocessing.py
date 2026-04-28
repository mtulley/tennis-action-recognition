

#Preprocess(INPUT_PATH, TRAIN_OUTPUT_PATH, VAL_OUTPUT_PATH,0.9)

#Preprocess THETIS tennis videos for TAO ActionRecognitionNet.

#Reads videos from class folders in INPUT_PATH.
#Splits each class into train / validation / test sets.
#Extracts each video into PNG frames using ffmpeg.
#Saves frames in TAO's expected folder format:

# print("finished and done")

import os
import subprocess
import random

INPUT_PATH = "/home/garywww/TAO/THETIS/VIDEO_RGB"
TRAIN_OUTPUT_PATH = "/home/garywww/TAO/datasets/fullframes2/train"
VAL_OUTPUT_PATH = "/home/garywww/TAO/datasets/fullframes2/val"
TEST_OUTPUT_PATH = "/home/garywww/TAO/datasets/fullframes2/test"


def clip_video(input_video_path, output_path, max_frames=None):
    #Extract frames from one video and save them as PNG images.
    os.makedirs(output_path, exist_ok=True)

    output_pattern = os.path.join(output_path, "%06d.png")

    cmd = [
    "ffmpeg",
    "-loglevel", "error",
    "-i", input_video_path,
    "-y"
]
    #Limit the number of extracted frames if requested.
    if max_frames is not None:
        cmd += ["-frames:v", str(max_frames)]

    cmd.append(output_pattern)

    # print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def preprocess(input_data_pwd, train_pwd, val_pwd, test_pwd, trainpercent, valpercent, max_frames=None):

    #Split videos into train / validation / test sets and extract frames.

    os.makedirs(train_pwd, exist_ok=True)
    os.makedirs(val_pwd, exist_ok=True)
    os.makedirs(test_pwd, exist_ok=True)
    labels = os.listdir(input_data_pwd)
    random.seed(33)

    for label in labels:
        label_input_path = os.path.join(input_data_pwd, label)

        if not os.path.isdir(label_input_path): #Ignore non folders
            continue

        label_output_path1 = os.path.join(train_pwd, label)
        label_output_path2 = os.path.join(val_pwd, label)
        label_output_path3 = os.path.join(test_pwd, label)
        os.makedirs(label_output_path1, exist_ok=True)
        os.makedirs(label_output_path2, exist_ok=True)
        os.makedirs(label_output_path3, exist_ok=True)

        #Get all video paths
        videos = [video for video in os.listdir(label_input_path) if video.lower().endswith(".avi") and os.path.isfile(os.path.join(label_input_path, video))]

        videos.sort()
        random.shuffle(videos)

        train_split_index = int(len(videos) * trainpercent)
        val_split_index = train_split_index + int(len(videos) * valpercent)

        train_videos = videos[:train_split_index]
        val_videos = videos[train_split_index:val_split_index]
        test_videos = videos[val_split_index:]
        
        #Getting the images from the videos for each set
        for video in train_videos:
            video_name = os.path.splitext(video)[0]
            input_video_path = os.path.join(label_input_path, video)
            output_video_path = os.path.join(label_output_path1, video_name, "rgb")

            clip_video(input_video_path, output_video_path, max_frames=max_frames)
            

        for video in val_videos:
            video_name = os.path.splitext(video)[0]
            input_video_path = os.path.join(label_input_path, video)
            output_video_path = os.path.join(label_output_path2, video_name, "rgb")

            clip_video(input_video_path, output_video_path, max_frames=max_frames)

        for video in test_videos:
            video_name = os.path.splitext(video)[0]
            input_video_path = os.path.join(label_input_path, video)
            output_video_path = os.path.join(label_output_path3, video_name, "rgb")

            clip_video(input_video_path, output_video_path, max_frames=max_frames)
            


preprocess(INPUT_PATH, TRAIN_OUTPUT_PATH, VAL_OUTPUT_PATH, TEST_OUTPUT_PATH, 0.8, 0.1)

print("finished and done")