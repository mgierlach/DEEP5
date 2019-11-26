import os
import torch
import torchvision
from torchvision.datasets.utils import download_url
import zipfile

train_path = 'train'
dl_file = 'dl2018-image-proj.zip'
dl_url = 'https://users.aalto.fi/mvsjober/misc/'

zip_path = os.path.join(train_path, dl_file)
if not os.path.isfile(zip_path):
    download_url(dl_url + dl_file, root=train_path, filename=dl_file, md5=None)

with zipfile.ZipFile(zip_path) as zip_f:
    zip_f.extractall(train_path)
    #os.unlink(zip_path)

import pandas as pd
import glob
from sklearn.preprocessing import OneHotEncoder
import platform


# Create an array with tuple(img, label) pairs from the annotations txt files.
files = glob.glob("./train/annotations/*")
labels = []
for name in files:
    try:
        with open(name) as f:
            if platform.system == "Windows":
                label = name.split("\\")[1].split(".")[0]
            else:
                label = name.split("/")[-1].split(".")[0]

            for line in f:
                labels.append((int(line.splitlines()[0]), label))

    except IOError as exc:  #Not sure what error this is
        if exc.errno != errno.EISDIR:
            raise

# Create dataframe with columns img, label and sort it by img and reset index without preserving old index values.
labels = pd.DataFrame(labels, columns=["img", "label"])
# One hot encoding
one_hot = pd.get_dummies(labels["label"])
labels = labels.join(one_hot).drop("label", axis="columns")
labels = labels.groupby("img", as_index=False).sum()
labels["img"] = labels["img"].astype("int32")
imgs_present = labels["img"].unique()
rows_to_add = []
for i in range(1, 20001):  # this takes a while sry
    if i in imgs_present:
        continue
    else:
        rows_to_add.append([i, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                            0])  # kinda ugly lol
df_to_append = pd.DataFrame(
    rows_to_add,
    columns=[
        "img", "baby", "bird", "car", "clouds", "dog", "female", "flower",
        "male", "night", "people", "portrait", "river", "sea", "tree"
    ])

labels = labels.append(df_to_append).sort_values("img").reset_index(drop=True)
print(labels.head())
labels.to_csv("./train/labels.csv", index=False)
