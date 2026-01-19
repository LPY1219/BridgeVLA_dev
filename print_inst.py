import pickle
import os

data_root = "/mnt/robot-rfm/user/lpy/data/RLBench/train/meat_off_grill/all_variations/episodes"

for i in range(100):
    episode_path = os.path.join(data_root, f"episode{i}", "variation_descriptions.pkl")

    if os.path.exists(episode_path):
        with open(episode_path, 'rb') as f:
            content = pickle.load(f)
        print(f"Episode {i}: {content}")
    else:
        print(f"Episode {i}: File not found")
        break