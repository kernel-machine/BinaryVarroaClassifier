import argparse
import json
import re
import os
import glob
from lib.VideoSegmenter import VideoSegmenter
import threading
import cv2
import shutil
from pathlib import Path
import random
import concurrent.futures
from multiprocessing.pool import ThreadPool

BOX_SIZE = 920

parser = argparse.ArgumentParser()
parser.add_argument(
    "--raw_videos", type=str, help="Folder with videos containing infested and free"
)
parser.add_argument("--label_file", type=str, help=".txt file containing label info")
parser.add_argument("--only_very_visible", default=False, action="store_true")
parser.add_argument(
    "--vs", default=0.3, type=float, help="Amount of the dataset used for validation"
)
parser.add_argument("--output", type=str, required=True)

args = parser.parse_args()

labels_file = open(args.label_file, "r")
labels = labels_file.readlines()
labels_file.close()

negative_frames = filter(lambda x: not json.loads(x)["varroa_visible"], labels)
negative_frames = list(negative_frames)

if args.only_very_visible:
    positive_frames = filter(lambda x: json.loads(x)["very_visibile"], labels)
else:
    positive_frames = filter(lambda x: json.loads(x)["varroa_visible"], labels)

positive_frames = list(positive_frames)


def get_video_id(x: str|dict):
    if type(x) is str:
        x = json.loads(x)
    x=x["video"].split(" ")[0]
    x = int(re.findall(r"\d+", x)[0])
    return x

positive_video_ids = set(map(get_video_id, positive_frames))
negative_video_ids = set(map(get_video_id, negative_frames))
negative_video_ids = negative_video_ids - positive_video_ids

#Remove videos where there are positive frames
negative_frames = filter(lambda x:get_video_id(x) in negative_video_ids, negative_frames)
negative_frames = list(negative_frames)
random.shuffle(negative_frames)

all_valid_frames = 2*min(len(negative_frames), len(positive_frames))
target_train_frames_amount = int((1-args.vs)*all_valid_frames)
target_val_frames_amount = all_valid_frames-target_train_frames_amount

"""
 @positive_frames -> frames from positive videos
 @negative_frames -> frames from negative videos
 @validation_split -> ratio between validation and train
 return
 frames_train, frames_val, such that:
    - validation_split is respected
    - frames_train contains 50% of varroa free and 50% of varroa infested
    - frames_val contains 50% of varroa free and 50% of varroa infested
    - The frames from the same video cannot be in both frames_train ans frames_val
"""
def split_frames(positive_frames:list[str], negative_frames:list[str], validation_split:float=0.3) -> tuple[list[str],list[str]]:
    total_amount_of_frames = min(len(positive_frames),len(negative_frames))*2
    target_train_amount = int((1-validation_split)*total_amount_of_frames)
    target_val_amount = total_amount_of_frames-target_train_amount

    val_positive_frames = []
    val_negative_frames = []
    train_positive_frames = []
    train_negative_frames = []

    frames_per_videos:dict[list[str]] = {}

    for frame in positive_frames+negative_frames:
        video_id = str(get_video_id(frame))
        if video_id not in frames_per_videos.keys():
            frames_per_videos[video_id]=[]
        frames_per_videos[video_id].append(frame)

    def get_frames_by_video(video_id:str, frame_class:bool) -> list[dict]:
        frames = []
        for frame in frames_per_videos[video_id]:
            jframe = json.loads(frame)
            if frame_class and (jframe["varroa_visible"] or jframe["very_visibile"]):
                frames.append(jframe)
            if not frame_class and not jframe["varroa_visible"] and not jframe["very_visibile"]:
                frames.append(jframe)
        return frames

    # Load positive frames in val set
    taken_videos = []
    print("Loading val positive frames")
    while len(val_positive_frames)<target_val_amount//2:
        for video_id in frames_per_videos.keys():
            # Check if video is already picked and if there are valid frames
            if video_id not in taken_videos and len(get_frames_by_video(video_id, True))>0:
                for f in get_frames_by_video(video_id, True):
                    val_positive_frames.append(f)
                taken_videos.append(video_id)
                break
    print(f"Added {len(val_positive_frames)} frames")


    print("Loading val negative frames")
    while len(val_negative_frames)<target_val_amount//2:
        for video_id in frames_per_videos.keys():
            # Check if video is already picked and if there are valid frames
            if video_id not in taken_videos and len(get_frames_by_video(video_id, False))>0:
                for f in get_frames_by_video(video_id, False):
                    val_negative_frames.append(f)
                taken_videos.append(video_id)
                break
    print(f"Added {len(val_negative_frames)} frames")

    print("Loading train positive frames")
    while len(train_positive_frames)<target_train_amount//2:
        videos_found = False
        for video_id in frames_per_videos.keys():
            # Check if video is already picked and if there are valid frames
            frames = get_frames_by_video(video_id, True)
            if video_id not in taken_videos and len(frames)>0:
                videos_found = True
                for f in frames:
                    train_positive_frames.append(f)
                taken_videos.append(video_id)
                break
        if not videos_found:
            break
    print(f"Added {len(train_positive_frames)} frames")

    print("Loading train negative frames")
    while len(train_negative_frames)<target_train_amount//2:
        videos_found = False
        for video_id in frames_per_videos.keys():
            # Check if video is already picked and if there are valid frames
            if video_id not in taken_videos and len(get_frames_by_video(video_id, False))>0:
                videos_found = True
                for f in get_frames_by_video(video_id, False):
                    train_negative_frames.append(f)
                taken_videos.append(video_id)
                break
        if not videos_found:
            break

    print(f"Added {len(train_negative_frames)} frames")


    print(len(val_positive_frames))
    print(len(val_negative_frames))
    print(len(train_positive_frames))
    print(len(train_negative_frames))

    print(f"Checking...")

    for val_frame in val_positive_frames+val_negative_frames:
        val_frame = val_frame["video"].split(" ")[0]
        val_frame = int(re.findall(r"\d+", val_frame)[0])
        for train_frame in train_positive_frames+train_negative_frames:
            train_frame = train_frame["video"].split(" ")[0]
            train_frame = int(re.findall(r"\d+", train_frame)[0])
            assert val_frame!=train_frame

    return train_positive_frames+train_negative_frames, val_positive_frames+val_negative_frames



def crop_frame(frame:str|dict, imgs_folder: str, destination_folder: Path):
    video_id = get_video_id(frame)
    if type(frame) is str:
        jlabel_frame = json.loads(frame)
    else:
        jlabel_frame = frame
    frame_id = jlabel_frame["id"]
    varroa_visible = int(
        jlabel_frame["very_visibile"] or jlabel_frame["varroa_visible"]
    )
    for img_file in glob.glob(imgs_folder):

        if (
            f"{video_id}_{frame_id}_" in os.path.basename(img_file)
            and os.path.basename(img_file)[len(str(video_id))] == "_"
        ):
            # Image found
            top_left = jlabel_frame["coord_1"]
            bottom_right = jlabel_frame["coord_2"]

            img = cv2.imread(img_file)

            if jlabel_frame["camera"] == "bottom":
                top_left = (top_left[0] - (7680 // 2), top_left[1])
                bottom_right = (bottom_right[0] - (7680 // 2), bottom_right[1])

            x1, y1 = top_left
            x2, y2 = bottom_right

            center = (x1+((x2-x1)//2), y1+((y2-y1)//2))
            x1, y1 = center[0]-(BOX_SIZE//2), center[1]-(BOX_SIZE//2)
            x2, y2 = center[0]+(BOX_SIZE//2), center[1]+(BOX_SIZE//2)
            height, width, _ = img.shape
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)

            # Crop the frame using slicing
            img = img[y1:y2, x1:x2]
            image_path = os.path.join(
                destination_folder,
                f"{video_id}_{frame_id}_{varroa_visible}.png",
            )
            cv2.imwrite(image_path, img)
            break

def crop_images(frames: list[str], imgs_folder: str, destination_folder: Path):
    for i, label_frame in enumerate(frames):
        print(
            f"Processing {i}/{len(frames)} from {os.path.basename(imgs_folder)}",
            end="\r" if i + 1 < len(frames) else "\n",
        )
        crop_frame(label_frame, imgs_folder, destination_folder)

print(f"Taking {len(negative_frames)} negative frames")
print(f"Taking {len(positive_frames)} positive frames")

destination_folder_train = Path(args.output,"train")
if destination_folder_train.exists():
    shutil.rmtree(destination_folder_train)
else:
    destination_folder_train.mkdir(parents=True)

destination_folder_val = Path(args.output,"val")
if destination_folder_val.exists():
    shutil.rmtree(destination_folder_val)
else:
    destination_folder_val.mkdir(parents=True)

frames_path = "/home/nonroot/dataset/labeled_unical/final_dataset/*/*.png"
train_frames, val_frames = split_frames(positive_frames,negative_frames,0.3)

print("Cropping frames")
# with ThreadPool(processes=2*os.cpu_count()) as tp:
#     tp.map(lambda x:crop_frame(x, frames_path, destination_folder_train), train_frames, chunksize=100)
#     tp.close()
#     tp.join()

# with ThreadPool(processes=2*os.cpu_count()) as tp:
#     tp.map(lambda x:crop_frame(x, frames_path, destination_folder_val), val_frames, chunksize=100)
#     tp.close()
#     tp.join()

    

t1 = threading.Thread(target=crop_images, args=(train_frames,frames_path, destination_folder_train,))
t2 = threading.Thread(target=crop_images, args=(val_frames,frames_path, destination_folder_val,))
t1.start()
t2.start()
t1.join()
t2.join()