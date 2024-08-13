import torch

path = "/mnt/hdd/davidwong/data/VLTVG/split/data/marinedet/marinedet_class_level_train_no_negative.pth"
data = torch.load(path)

for d in data[:10]:
    print(d)
