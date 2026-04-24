import pickle

# Path to the desired pickle file
PKL_PTH = "./dataset/pose_classification/test_label.pkl"

with open(PKL_PTH, "rb") as f:
    data = pickle.load(f)

print("TYPE:", type(data))

if isinstance(data, dict):
    print("KEYS:", data.keys())
    for k, v in data.items():
        try:
            print(k, "->", type(v), "len:", len(v))
        except:
            print(k, "->", type(v))

elif isinstance(data, (list, tuple)):
    print("LEN:", len(data))
    print("FIRST 5:", data[:5])

else:
    print(data)