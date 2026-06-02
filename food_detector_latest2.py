import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import requests

from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms
from torchvision.ops import box_iou
from torch.optim.lr_scheduler import CosineAnnealingLR
from collections import defaultdict

# GLOBALS
IMG_DIR   = r"C:\Users\sevvl\Desktop\turkish_food_dataset_\all_images_new"
LABEL_DIR = r"C:\Users\sevvl\Desktop\turkish_food_dataset_\labels22"

BATCH_SIZE = 4
EPOCHS = 20
IMG_SIZE = 512
LR = 0.005

CONF_THRESHOLD = 0.45
NMS_IOU = 0.35
VAL_RATIO = 0.15
IOU_THRESHOLD = 0.50

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

model_global = None
val_loader_global = None

# FOOD MAP
FOOD_MAP = {
    0:"baklagil",1:"ekmek",2:"pilav",3:"kirmizi et",
    4:"salata",5:"balik",6:"patates",7:"tavuk",
    8:"sebze",9:"makarna",10:"corba",11:"zeytinyagli",
    12:"yumurta",13:"yogurt",14:"meyve",15:"manti",
    16:"pide",17:"fastfood",18:"lahmacun",19:"tatli"
}
NUM_CLASSES = len(FOOD_MAP) + 1

# DATASET
def imread(path):
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)

def yolo_to_pixel(x,y,w,h,img_w,img_h):
    x1 = (x - w/2)*img_w
    y1 = (y - h/2)*img_h
    x2 = (x + w/2)*img_w
    y2 = (y + h/2)*img_h
    return [x1,y1,x2,y2]

class FoodDataset(Dataset):
    def __init__(self, img_dir, label_dir):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.images = []

        for img in os.listdir(img_dir):
            lbl = os.path.join(label_dir, img.replace(".jpg",".txt"))
            if os.path.exists(lbl):
                self.images.append(img)

        self.tf = transforms.ToTensor()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img = imread(os.path.join(self.img_dir,img_name))
        img = cv2.resize(img,(IMG_SIZE,IMG_SIZE))
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

        label_path = os.path.join(self.label_dir,img_name.replace(".jpg",".txt"))

        boxes, labels = [], []

        with open(label_path) as f:
            for line in f:
                c,x,y,w,h = map(float,line.split())
                x1,y1,x2,y2 = yolo_to_pixel(x,y,w,h,IMG_SIZE,IMG_SIZE)

                boxes.append([x1,y1,x2,y2])
                labels.append(int(c)+1)

        boxes = torch.tensor(boxes,dtype=torch.float32)
        labels = torch.tensor(labels,dtype=torch.int64)

        target = {
            "boxes":boxes,
            "labels":labels,
            "area":(boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1]),
            "iscrowd":torch.zeros(len(labels))
        }

        return self.tf(img), target

def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    return tuple(zip(*batch))

# MODEL
def get_model():
    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    in_feat = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_feat, NUM_CLASSES)
    return model

# TRAIN
def train():
    global model_global, val_loader_global

    dataset = FoodDataset(IMG_DIR,LABEL_DIR)

    n_val = int(len(dataset)*VAL_RATIO)
    n_train = len(dataset)-n_val

    train_set,val_set = random_split(dataset,[n_train,n_val])

    train_loader = DataLoader(train_set,batch_size=BATCH_SIZE,shuffle=True,collate_fn=collate_fn)
    val_loader = DataLoader(val_set,batch_size=BATCH_SIZE,collate_fn=collate_fn)

    model = get_model().to(device)

    optimizer = torch.optim.SGD(model.parameters(),lr=LR,momentum=0.9)
    scheduler = CosineAnnealingLR(optimizer,EPOCHS)

    for epoch in range(EPOCHS):
        model.train()
        total=0

        for imgs,targets in train_loader:
            imgs = [i.to(device) for i in imgs]
            targets = [{k:v.to(device) for k,v in t.items()} for t in targets]

            loss_dict = model(imgs,targets)
            loss = sum(loss_dict.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total += loss.item()

        scheduler.step()

        print(f"Epoch {epoch} Loss:{total:.4f}")

    torch.save(model.state_dict(),"model.pth")

    model_global = model
    val_loader_global = val_loader

    print("Training done")

# LOAD

def load_model():
    model = get_model().to(device)
    model.load_state_dict(torch.load("model.pth",map_location=device))
    model.eval()
    return model

# PREDICT
def predict(model,img_path):
    img = imread(img_path)
    img = cv2.resize(img,(IMG_SIZE,IMG_SIZE))
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    t = transforms.ToTensor()(img).to(device)

    with torch.no_grad():
        out = model([t])[0]

    for box,label,score in zip(out["boxes"],out["labels"],out["scores"]):
        if score < CONF_THRESHOLD:
            continue

        x1,y1,x2,y2 = map(int,box.cpu())
        name = FOOD_MAP.get(label.item()-1,"unknown")

        cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.putText(img,f"{name} {score:.2f}",(x1,y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)

    plt.imshow(img)
    plt.show()

# BENCHMARK
def benchmark(model,loader):
    model.eval()
    t0 = time.time()

    for i,(imgs,_) in enumerate(loader):
        imgs = [i.to(device) for i in imgs]
        with torch.no_grad():
            model(imgs)
        if i>20:
            break

    return time.time()-t0

# MAIN
if __name__=="__main__":
    print("1-Train")
    print("2-Predict")

    c = input()

    if c=="1":
        train()

    if c=="2":
        m = load_model()
        p = input("image path:")
        predict(m,p)