import argparse
import os
import json
import re

parser = argparse.ArgumentParser()
parser.add_argument("--label_file", type=str, required=True)
parser.add_argument("--folder_with_frames", type=str, required=True)
args = parser.parse_args()

frames = os.listdir(args.folder_with_frames)
f = open(args.label_file, "r")
lines = f.readlines()
f.close()

new_file = open("out.txt","w")
very_very = 0

print(f"Found {len(frames)} frames")

for line_index,line in enumerate(lines):
    line = json.loads(line)

    if "infested" in line["video"]:
        label_frame_id = int(re.findall(r'\d+',line["id"])[0])
        label_video_id = int(os.path.basename(line["video"]).split(" ")[0])

        print(f"Processing line {line_index} of {len(lines)}", end="\r" if line_index+1<len(lines) else "\n")

        assert line["id"]==f"frame{label_frame_id}", f"Frame id doesn't match {line['id']} = frame{label_frame_id}"
        assert f"varroa_infested/{label_video_id}" in line["video"], f"video id doesn't match {label_video_id} {line['video']} "

        for frame in frames:
            video_id = int(frame.split("_")[0])
            frame_id = frame.split("_")[1]
            frame_id = int(re.findall(r'\d+',frame_id)[0])

            assert f"{video_id}_frame{frame_id}" in frame, "Frame doesn't match"
            
            if label_frame_id == frame_id and label_video_id == video_id:
                line["very_visibile"] = True
                very_very+=1
                break
            else:
                line["very_visibile"] = False
    else:
        line["very_visibile"] = False

    line["varroa_visible"] = line["varroa_visible"]=="yes"
    new_file.write(json.dumps(line))
    new_file.write("\n")

assert very_very == len(frames), "Output frame number doesn't match"
new_file.close()

        

