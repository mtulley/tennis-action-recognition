from pathlib import Path
import json
import cv2

VIDEOS_ROOT = Path("/workspace/dataset/videos_mp4")
OUTPUT_ROOT = Path("/workspace/dataset/skeletons_json")

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

for class_dir in sorted(VIDEOS_ROOT.iterdir()):
    if not class_dir.is_dir():
        continue

    label = class_dir.name
    out_class_dir = OUTPUT_ROOT / label
    out_class_dir.mkdir(parents=True, exist_ok=True)

    videos = sorted(class_dir.rglob("*.mp4"))

    for video_file in videos:
        output_file = out_class_dir / f"{video_file.stem}.json"

        cap = cv2.VideoCapture(str(video_file))
        if not cap.isOpened():
            print(f"Failed to open {video_file}")
            continue

        data = {
            "label": label,
            "video_file": video_file.name,
            "frames": []
        }

        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_record = {
                "frame_idx": frame_idx,
                "persons": []
            }

            data["frames"].append(frame_record)
            frame_idx += 1

        cap.release()

        with open(output_file, "w") as f:
            json.dump(data, f, indent=2)

        print(f"Wrote {output_file} with {frame_idx} frames")