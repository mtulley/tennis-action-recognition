from pathlib import Path
import json

VIDEOS_ROOT = Path("/workspace/dataset/videos_mp4")
OUTPUT_ROOT = Path("/workspace/dataset/skeletons_json")

print("VIDEOS_ROOT exists:", VIDEOS_ROOT.exists())
print("OUTPUT_ROOT:", OUTPUT_ROOT)

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

for class_dir in sorted(VIDEOS_ROOT.iterdir()):
    print("Found entry:", class_dir)
    if not class_dir.is_dir():
        print("  skipping, not a dir")
        continue

    label = class_dir.name
    out_class_dir = OUTPUT_ROOT / label
    out_class_dir.mkdir(parents=True, exist_ok=True)
    print("  label:", label)

    videos = sorted(class_dir.glob("*.mp4"))
    print("  mp4 count:", len(videos))

    for video_file in videos:
        output_file = out_class_dir / f"{video_file.stem}.json"
        data = {
            "label": label,
            "video_file": video_file.name,
            "frames": []
        }
        with open(output_file, "w") as f:
            json.dump(data, f, indent=2)
        print("  wrote:", output_file)