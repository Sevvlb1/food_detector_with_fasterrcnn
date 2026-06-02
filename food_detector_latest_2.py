import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms
from torch.optim.lr_scheduler import CosineAnnealingLR  
from torchvision.ops import box_iou
from collections import defaultdict
import requests

IMG_DIR   = r"C:\Users\sevvl\Desktop\turkish_food_dataset_\all_images_new"
LABEL_DIR = r"C:\Users\sevvl\Desktop\turkish_food_dataset_\labels22"

BATCH_SIZE  = 2      
EPOCHS      = 35      
IMG_SIZE    = 512    
LR          = 0.001   

CONF_THRESHOLD = 0.35  # false positivesleri azaltmak için 0.22 confu fahil etmez yanş
NMS_IOU        = 0.45  # aynı obhje için birden fazla bb çizmememsş için
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15
IOU_THRESHOLD  = 0.50

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

FOOD_MAP = {
    0: "baklagil", 1: "ekmek",    2: "pilav",     3: "kirmizi et",
    4: "salata",   5: "balik",    6: "patates",   7: "tavuk",
    8: "sebze",    9: "makarna",  10: "corba",    11: "zeytinyagli",
    12: "yumurta", 13: "yogurt",  14: "meyve",    15: "manti",
    16: "pide",    17: "fastfood",18: "lahmacun", 19: "tatli"
}
NUM_CLASSES = len(FOOD_MAP) + 1

API_KEY = "WsTZaCKNfS2y6stoS2Ydx52yfpWVoPMWABrnF1VZ"
kcal_cache = {}

#usdaya yemek adı yollar
def get_kcal(food_name):
    url = "https://api.nal.usda.gov/fdc/v1/foods/search"
    params = {"query": food_name, "api_key": API_KEY, "pageSize": 1}
    r = requests.get(url, params=params, timeout=3)
    data = r.json()
    try:
        food = data["foods"][0]
        for n in food["foodNutrients"]:
            if "Energy" in n["nutrientName"]:
                return n["value"]
    except:
        return None

DEFAULT_KCAL = 150

# api çöktüğünde default atar güvemlik katmanı
def safe_kcal(food_name):
    if food_name in kcal_cache:
        return kcal_cache[food_name]
    try:
        kcal = get_kcal(food_name)
    except:
        kcal = None
    if kcal is None:
        kcal = DEFAULT_KCAL
    kcal_cache[food_name] = kcal
    return kcal

API_MAP = {
    "baklagil": "lentils", "ekmek": "bread", "pilav": "rice cooked",
    "kirmizi et": "beef cooked", "salata": "green salad", "balik": "grilled fish",
    "patates": "boiled potato", "tavuk": "chicken breast cooked",
    "sebze": "mixed vegetables", "makarna": "pasta cooked", "corba": "soup",
    "zeytinyagli": "vegetable with olive oil", "yumurta": "boiled egg",
    "yogurt": "plain yogurt", "meyve": "fresh fruit", "manti": "dumplings",
    "pide": "flatbread", "fastfood": "hamburger", "lahmacun": "turkish pizza",
    "tatli": "dessert"
}

PORTION_G = {
    "baklagil": 91, "ekmek": 40, "pilav": 170, "kirmizi et": 150,
    "salata": 300, "balik": 200, "patates": 80, "tavuk": 250,
    "sebze": 100, "makarna": 90, "corba": 150, "zeytinyagli": 150,
    "yumurta": 50, "yogurt": 160, "meyve": 120, "manti": 100,
    "pide": 170, "fastfood": 190, "lahmacun": 150, "tatli": 160
}
DEFAULT_PORTION = 150


def get_nutrition(food_name):
    url = "https://api.nal.usda.gov/fdc/v1/foods/search"

    params = {
        "query": food_name,
        "api_key": API_KEY,
        "pageSize": 1
    }

    r = requests.get(url, params=params, timeout =5)
    data = r.json()

    food = data["foods"][0]
     
    result={
        "kcal":0,
        "protein":0,
        "carb":0,
        "fat":0
    }

    for n in food["foodNutrients"]:
        name = n.get("nutrientName", "")
        value = n.get("value", 0)

        if "Energy" in name:
            result["kcal"] = value

        elif "Protein" in name:
            result["protein"] = value

        elif "Carbohydrate" in name:
            result["carb"] = value
        
        elif "Total lipid" in name:
            result["fat"] = value

    return result 

nutrition_cache = {}
def safe_nutrition(food_name):
    if food_name in nutrition_cache:
        return nutrition_cache[food_name]
    
    try:
        nutrition = get_nutrition(food_name)
    except:
        nutrition = {
            "kcal": 150,
            "protein": 5,
            "carb": 20,
            "fat": 5
        }
    nutrition_cache[food_name] = nutrition
    return nutrition


def estimate_calories(yolo_class_id, box, img_size=512):

    food_name = FOOD_MAP.get(yolo_class_id, None)
    if food_name is None:
        food_name = f"unknown_{yolo_class_id}"

    query_name    = API_MAP.get(food_name, food_name)
    nutrition = safe_nutrition(query_name)
    kcal_per_100g = nutrition["kcal"]
    protein_per_100g = nutrition["protein"]
    carb_per_100g = nutrition["carb"]
    fat_per_100g = nutrition["fat"]

    portion_g = PORTION_G.get(food_name, DEFAULT_PORTION)

    x1, y1, x2, y2 = box

    # BOX SIZE
    box_w = max(1.0, x2 - x1)
    box_h = max(1.0, y2 - y1)

    box_area = box_w * box_h
    frame_area = img_size * img_size

    area_ratio = box_area / frame_area

    # OBJECT CENTER
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    img_center_x = img_size / 2.0
    img_center_y = img_size / 2.0

    # merkeze uzaklık
    dist = np.sqrt(
        (cx - img_center_x) ** 2 +
        (cy - img_center_y) ** 2
    )

    max_dist = np.sqrt(
        img_center_x ** 2 +
        img_center_y ** 2
    )

    center_score = 1.0 - (dist / max_dist)

    # PERSPECTIVE CORRECTION
    # merkeze yakın objeler biraz daha güvenilir
    perspective_factor = 0.75 + (center_score * 0.5)

    corrected_ratio = area_ratio * perspective_factor

    # SCALE NORMALIZATION
    reference_ratio = 0.18

    scale = corrected_ratio / reference_ratio

    # aşırı büyümeyi engelle
    scale = np.clip(scale, 0.55, 1.85)

    # FINAL WEIGHT
    weight_g = portion_g * scale

    # çok küçük bbox ama yüksek close-up durumlarını dengele
    if area_ratio < 0.04:
        weight_g *= 0.90

    # aşırı büyük bbox için fren
    if area_ratio > 0.35:
        weight_g *= 0.82

    calories = (weight_g / 100.0) * kcal_per_100g
    scale = weight_g / 100.0
    protein = protein_per_100g * scale
    carb = carb_per_100g * scale
    fat = fat_per_100g * scale

    return {
        "food" : food_name,
        "weight" : round(weight_g, 1),
        "kcal" : round(calories, 1),
        "protein" : round(protein, 1),
        "carb" : round(carb, 1),
        "fat" : round(fat, 1)

    }
        
    

def imread(path):
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR) #tr karakterli pathler için

#yolo formatını pixele (x1, y1) gibi çeviriyor 
def yolo_to_pixel(x, y, w, h, img_w, img_h):
    x1 = (x - w / 2) * img_w
    y1 = (y - h / 2) * img_h
    x2 = (x + w / 2) * img_w
    y2 = (y + h / 2) * img_h
    return [x1, y1, x2, y2]


class FoodDataset(Dataset):
    def __init__(self, img_dir, label_dir, augment=False):
        self.img_dir   = img_dir
        self.label_dir = label_dir
        self.augment   = augment  
        if augment:
            self.tf = transforms.Compose([
                transforms.ToPILImage(),
                #modelin ışık kontrast renk değişimlerine dayanıklı olmasını sağlıyor 
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
                transforms.RandomGrayscale(p=0.05),
                transforms.ToTensor(),
            ])
        else:
            self.tf = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),
            ])

        all_images = os.listdir(img_dir)
        self.images = []
        skipped = 0

        for img_name in all_images:
            label_path = self._label_path(img_name)
            if not os.path.exists(label_path):
                skipped += 1
                continue
            with open(label_path) as f:
                lines = [l.strip() for l in f if l.strip()]
            if not lines:
                skipped += 1
                continue
            self.images.append(img_name)

        print(f"Dataset: {len(self.images)} valid | {skipped} skipped")

    def _label_path(self, img_name):
        base = os.path.splitext(img_name)[0]
        return os.path.join(self.label_dir, base + ".txt")

    def __len__(self):
        return len(self.images)

    #datasetten 1 sample çekiyor
    def __getitem__(self, idx):
        img_name   = self.images[idx]
        img_path   = os.path.join(self.img_dir, img_name)
        label_path = self._label_path(img_name)

        img = imread(img_path)
        if img is None:
            return self.__getitem__((idx + 1) % len(self.images))

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) #512x512
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        #data augmentation (veri arttırımı)
        if self.augment and np.random.rand() > 0.5:
            img = np.fliplr(img).copy()
            flipped = True
        else:
            flipped = False

        boxes, labels = [], []
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls, x, y, bw, bh = map(float, parts)

                #flip sonrası bb yeri
                if flipped:
                    x = 1.0 - x

                x1, y1, x2, y2 = yolo_to_pixel(x, y, bw, bh, IMG_SIZE, IMG_SIZE)
                x1 = max(0.0, min(x1, IMG_SIZE - 1))
                y1 = max(0.0, min(y1, IMG_SIZE - 1))
                x2 = max(0.0, min(x2, IMG_SIZE - 1))
                y2 = max(0.0, min(y2, IMG_SIZE - 1))

                if x2 <= x1 or y2 <= y1:
                    continue

                boxes.append([x1, y1, x2, y2])
                labels.append(int(cls) + 1)

        if len(boxes) == 0:
            return None

        boxes   = torch.as_tensor(boxes,  dtype=torch.float32)
        labels  = torch.as_tensor(labels, dtype=torch.int64)
        areas   = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        iscrowd = torch.zeros(len(labels), dtype=torch.int64)

        return self.tf(img), {
            "boxes": boxes, "labels": labels,
            "area": areas,  "iscrowd": iscrowd,
        }


#her image farklı bb içerdiği için custom collate gerekiyor 
def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    return tuple(zip(*batch))



#MODEL 
def get_model(num_classes, fine_tune=True):

    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights)

    # transfer learning
    if not fine_tune:
        for param in model.backbone.parameters():
            param.requires_grad = False

    in_features = model.roi_heads.box_predictor.cls_score.in_features

    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features,
        num_classes
    )

    model.roi_heads.score_thresh = 0.40
    model.roi_heads.nms_thresh = 0.35
    model.roi_heads.detections_per_img = 20

    return model


#val loss
def compute_val_loss(model, loader):
    model.train()
    total, n = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue
            images, targets = batch
            if len(images) == 0:
                continue
            images  = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            valid   = [(img, tgt) for img, tgt in zip(images, targets)
                       if tgt["boxes"].numel() > 0]
            if not valid:
                continue
            images, targets = zip(*valid)
            try:
                loss_dict = model(list(images), list(targets))
                total += sum(loss_dict.values()).item()
                n += 1
            except Exception as e:
                print(f"  [val_loss] skipped batch: {e}")
    model.eval()
    return total / n if n > 0 else float("nan")

#mAP temeli /prediction
def match_predictions(pred_boxes, pred_labels, pred_scores,
                       gt_boxes, gt_labels,
                       iou_thresh=IOU_THRESHOLD,
                       score_thresh=CONF_THRESHOLD):
    results = []
    keep = pred_scores >= score_thresh
    pred_boxes, pred_labels, pred_scores = pred_boxes[keep], pred_labels[keep], pred_scores[keep]

    if len(gt_boxes) == 0:
        for s, lbl in zip(pred_scores, pred_labels):
            results.append((float(s), False, int(lbl), -1))
        return results
    if len(pred_boxes) == 0:
        return results

    iou_matrix = box_iou(
        torch.as_tensor(pred_boxes, dtype=torch.float32),
        torch.as_tensor(gt_boxes,   dtype=torch.float32)
    ).numpy()

    matched_gt = set()
    for pi in np.argsort(-pred_scores):
        best_iou_idx = int(np.argmax(iou_matrix[pi]))
        best_iou     = iou_matrix[pi, best_iou_idx]
        if best_iou >= iou_thresh and best_iou_idx not in matched_gt:
            matched_gt.add(best_iou_idx)
            is_tp    = (pred_labels[pi] == gt_labels[best_iou_idx])
            gt_label = int(gt_labels[best_iou_idx])
        else:
            is_tp, gt_label = False, -1
        results.append((float(pred_scores[pi]), is_tp, int(pred_labels[pi]), gt_label))
    return results

#mAP hesabı kutu doğru yere çizilmiş mi sınıf dopru mu hesaplae 
def compute_map(all_results, num_classes):
    gt_count = defaultdict(int)
    for _, _, _, gt_lbl in all_results:
        if gt_lbl != -1:
            gt_count[gt_lbl] += 1

    by_class = defaultdict(list)
    for score, is_tp, pred_lbl, _ in all_results:
        by_class[pred_lbl].append((score, is_tp))

    ap_dict = {}
    for cls in range(1, num_classes):          # ← tek döngü
        preds = sorted(by_class.get(cls, []), key=lambda x: -x[0])
        n_gt  = gt_count.get(cls, 0)
        if n_gt == 0:
            ap_dict[cls] = 0.0
            continue
        tp_cum    = np.cumsum([1 if tp else 0 for _, tp in preds])
        fp_cum    = np.cumsum([0 if tp else 1 for _, tp in preds])
        precision = tp_cum / (tp_cum + fp_cum + 1e-9)
        recall    = tp_cum / (n_gt + 1e-9)
        ap = sum(
            precision[recall >= t].max() if (recall >= t).any() else 0.0
            for t in np.linspace(0, 1, 11)
        ) / 11.0
        ap_dict[cls] = ap

    mAP = float(np.mean(list(ap_dict.values()))) if ap_dict else 0.0
    return ap_dict, mAP          # ← return döngü dışında


def compute_metrics(all_results):
    tp = sum(1 for _, is_tp, _, _ in all_results if is_tp)

    fp = sum(
        1 for _, is_tp, pred_lbl, gt_lbl in all_results
        if not is_tp and gt_lbl == -1
    )

    fn = sum(
        1 for _, is_tp, pred_lbl, gt_lbl in all_results
        if not is_tp and gt_lbl != -1
    )

    precision = tp/ (tp + fp + 1e-9)
    recall = tp/ (tp + fn + 1e-9)

    f1 = 2* precision*recall / (precision+recall+ 1e-9)
    accuracy = tp / (tp+fp+fn+1e-9)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy
    }



#hangi yemek hangi yemekle karışıyor 
def build_confusion_matrix(all_results, num_classes):
    mat = np.zeros((num_classes, num_classes), dtype=int)
    for _, is_tp, pred_lbl, gt_lbl in all_results:
        p = min(pred_lbl, num_classes - 1)
        g = min(gt_lbl if gt_lbl != -1 else 0, num_classes - 1)
        mat[g, p] += 1
    return mat


@torch.no_grad()
def evaluate_val_set(model, loader):
    model.eval()
    all_results = []
    for batch in loader:
        if not batch or len(batch[0]) == 0:
            continue
        images, targets = batch
        images  = [img.to(device) for img in images]
        outputs = model(images)
        for output, target in zip(outputs, targets):
            res = match_predictions(
                output["boxes"].cpu().numpy(),
                output["labels"].cpu().numpy(),
                output["scores"].cpu().numpy(),
                target["boxes"].numpy(),
                target["labels"].numpy()
            )
            all_results.extend(res)
    return all_results

#cf ap graph ve loss curve çiziyor 
def plot_confusion_matrix(conf_mat, class_names, save_path="confusion_matrix.png"):
    cm = conf_mat[1:21,1:21]
    row_sums = cm.sum(axis=1, keepdims=True).clip(min=1)
    cm_norm  = cm / row_sums
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(30,15)
    )
    sns.heatmap(cm, ax=axes[0], annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, linewidths=0.3)
    axes[0].set_title("Confusion Matrix – Raw Counts", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("Ground Truth")
    axes[0].tick_params(axis="x", rotation=45); axes[0].tick_params(axis="y", rotation=0)
    sns.heatmap(cm_norm, ax=axes[1], annot=True, fmt=".2f", cmap="Greens",
                xticklabels=class_names, yticklabels=class_names,
                linewidths=0.3, vmin=0, vmax=1)
    axes[1].set_title("Confusion Matrix – Row-Normalised (Recall)", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("Ground Truth")
    axes[1].tick_params(axis="x", rotation=45); axes[1].tick_params(axis="y", rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.show()
    print(f"Confusion matrix saved → {save_path}")


def plot_per_class_ap(ap_dict, class_names, save_path="per_class_ap.png"):
    all_cls_ids = list(range(1, len(class_names) + 1))
    ap_values   = [ap_dict.get(c, 0.0) for c in all_cls_ids]
    names       = [class_names[c - 1] for c in all_cls_ids]

    colors = []
    for v in ap_values:
        if v >= 0.6:
            colors.append("#2ecc71")
        elif v >= 0.3:
            colors.append("#f39c12")
        else:
            colors.append("#e74c3c")

    fig, ax = plt.subplots(figsize=(16, 6))
    bars = ax.bar(names, ap_values, color=colors, edgecolor="white", width=0.6)
    ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=3)

    mAP = np.mean(ap_values)
    ax.axhline(mAP, color="navy", linewidth=1.5, linestyle="--", label=f"mAP = {mAP:.3f}")

    ax.set_ylim(0, 1.10)
    ax.set_ylabel("Average Precision (AP@50)")
    ax.set_title("Per-Class Average Precision – Test Set", fontsize=13, fontweight="bold")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.show()
    print(f"Per-class AP saved → {save_path}")

#train test val graph
def plot_loss_curves(train_hist, val_hist, save_path="loss_curves.png"):
    epochs = range(1, len(train_hist) + 1)
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        epochs,
        train_hist,
        marker="o",
        linewidth=2,
        label="Train Loss"
    )

    ax.plot(
        epochs,
        val_hist,
        marker="s",
        linewidth=2,
        label="Validation Loss"
    )


    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(
        "Training and Validation Loss",
        fontsize=13,
        fontweight="bold"
    )
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=130)
    plt.show()
    print(f"Loss curves saved → {save_path}")




def plot_metric_curves(
    precision_hist,
    recall_hist,
    f1_hist,
    accuracy_hist,
    save_path="metric_curves.png"
):

    epochs = range(1, len(f1_hist) + 1)

    fig, ax = plt.subplots(figsize=(12, 7))

    # VALIDATION
    ax.plot(epochs, precision_hist, marker="o", label="Val Precision")
    ax.plot(epochs, recall_hist, marker="s", label="Val Recall")
    ax.plot(epochs, f1_hist, marker="^", label="Val F1")
    ax.plot(epochs, accuracy_hist, marker="d", label="Val Accuracy")


    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.set_title("Validation vs Test Metrics")

    ax.set_ylim(0, 1)

    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=130)
    plt.show()

    print(f"Metric curves saved → {save_path}")




#TRAIN 
def train():
    full_dataset = FoodDataset(IMG_DIR, LABEL_DIR, augment=False)

    # YENİ — stratified split
    from collections import defaultdict

# Her örnek için dominant class etiketini bul
    def get_dominant_label(dataset, idx):
        result = dataset[idx]
        if result is None:
            return -1
        _, target = result
        labels = target["labels"].tolist()
        if not labels:
            return -1
        return max(set(labels), key=labels.count)

    # Sınıfa göre indeksleri grupla
    class_to_indices = defaultdict(list)
    for i in range(len(full_dataset)):
        lbl = get_dominant_label(full_dataset, i)
        if lbl != -1:
            class_to_indices[lbl].append(i)

    train_idx, val_idx, test_idx = [], [], []
    rng = np.random.default_rng(42)

    for lbl, indices in class_to_indices.items():
        indices = rng.permutation(indices).tolist()
        n = len(indices)
        n_tr = max(1, int(n * TRAIN_RATIO))
        n_vl = max(1, int(n * VAL_RATIO))
        # Kalan tümü test'e (minimum 1 garantili)
        n_te = n - n_tr - n_vl
        if n_te < 1:          # çok az örnek varsa val'dan al
            n_vl = max(1, n_vl - 1)
            n_te = n - n_tr - n_vl
        train_idx += indices[:n_tr]
        val_idx   += indices[n_tr:n_tr + n_vl]
        test_idx  += indices[n_tr + n_vl:]

    train_set = torch.utils.data.Subset(full_dataset, train_idx)
    val_set   = torch.utils.data.Subset(full_dataset, val_idx)
    test_set  = torch.utils.data.Subset(full_dataset, test_idx)

    n_train, n_val, n_test = len(train_set), len(val_set), len(test_set)

    #test için tüm classlaer 
    from collections import Counter

    test_counter = Counter()

    for img, target in test_set:

        for lbl in target["labels"]:
            test_counter[int(lbl)] += 1

    print("\nTEST CLASS COUNTS")

    for cls in range(1, NUM_CLASSES):

        print(
            FOOD_MAP[cls-1],
            test_counter.get(cls,0)
        )

    print(f"Train: {n_train}")
    print(f"Val  : {n_val}")
    print(f"Test : {n_test}")

    #train_set.dataset.augment = False  
    class AugmentedSubset(torch.utils.data.Dataset):
        def __init__(self, subset):
            self.subset = subset
            self.aug_tf = transforms.Compose([
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
                transforms.RandomGrayscale(p=0.05),
                transforms.ToTensor(),
            ])

        def __len__(self):
            return len(self.subset)

        def __getitem__(self, idx):                          
            result = self.subset[idx]
            if result is None:
                return None
            img, target = result

            pil = transforms.ToPILImage()(img)              
            img = self.aug_tf(pil)                          

            if np.random.rand() > 0.5:
                img = torch.flip(img, [2])
                boxes = target["boxes"].clone()
                x1 = IMG_SIZE - target["boxes"][:, 2]
                x2 = IMG_SIZE - target["boxes"][:, 0]
                boxes[:, 0] = x1
                boxes[:, 2] = x2
                
                target = {**target, "boxes": boxes}

            return img, target
    
    augmented_train = AugmentedSubset(train_set)

    print(f"Split → Train: {n_train} | Test: {n_test} | Val: {n_val}")

    train_loader = DataLoader(
        augmented_train, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collate_fn, num_workers=0,
    )
    val_loader = DataLoader(
        val_set, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate_fn, num_workers=0,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    model = get_model(NUM_CLASSES, fine_tune=False).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=LR, momentum=0.9, weight_decay=0.0005
    )
    # CosineAnnealingLR — smoothly decays LR, much better than StepLR for food detection
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

    train_loss_history, val_loss_history = [], []
    precision_history=[]
    recall_history=[]
    f1_history=[]
    accuracy_history= []


    #best_val_loss = float("inf")
    best_f1 = 0.0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss, n_batches = 0, 0

        #5 epochtran sonra backbone açılıyor 
        if epoch == 5:
            print("Fine-tuning backbone enabled.")

            for param in model.backbone.parameters():
                param.requires_grad = True

            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=LR * 0.1,
                momentum=0.9,
                weight_decay=0.0005
            )
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max= EPOCHS - epoch,
                eta_min=1e-5
            )


        for batch in train_loader:
            if batch is None:
                continue
            images, targets = batch
            if len(images) == 0:
                continue
            images  = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            try:
                #loss_dict classfication loss bb regression loss rpn loss
                loss_dict = model(images, targets)
                losses    = sum(loss_dict.values())
                optimizer.zero_grad()
                losses.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                total_loss += losses.item()
                n_batches  += 1
            except Exception as e:
                print(f"  Skipped batch: {e}")
                continue

        #her epoch sonunda LR düşüyor 
        scheduler.step()
        torch.cuda.empty_cache()

        avg_train = total_loss / max(n_batches, 1)
        avg_val   = compute_val_loss(model, val_loader)

        train_loss_history.append(avg_train)
        val_loss_history.append(avg_val)

        val_results = evaluate_val_set(model, val_loader)
        metrics = compute_metrics(val_results)

        precision_history.append(metrics["precision"])
        recall_history.append(metrics["recall"])
        f1_history.append(metrics["f1"])
        accuracy_history.append(metrics["accuracy"])

        

        print(
            f"Epoch {epoch:>2}/{EPOCHS} | "
            f"Train Loss: {avg_train:.4f} | "
            f"Val Loss: {avg_val:.4f} | "
            f"Precision: {metrics['precision']:.4f} | "
            f"Recall: {metrics['recall']:.4f} | "
            f"F1: {metrics['f1']:.4f} | "
            f"Accuracy: {metrics['accuracy']:.4f} | "
            f"LR: {scheduler.get_last_lr()[0]:.6f}"
        )

        # en düşük val loss saklanıyor 
        '''if not np.isnan(avg_val) and avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), "best_food_model.pth")
            print(f"Best model saved (val loss: {best_val_loss:.4f})")'''
        
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            torch.save(model.state_dict(), "best_food_model.pth")
            print(f"Best model saved (F1: {best_f1:.4f})")


    # son model 
    torch.save(model.state_dict(), "modelfoodplus3.pth")
    print("Model saved → modelfoodplus3.pth")

    print("\nRunning final validation evaluation…")

    model = get_model(NUM_CLASSES).to(device)
    model.load_state_dict(
    torch.load(
        "best_food_model.pth",
        map_location=device,
        weights_only=True
        )
    )
    model.eval()





    all_results  = evaluate_val_set(model, val_loader)
    ap_dict, mAP = compute_map(all_results, NUM_CLASSES)
    print(f"\n  mAP@{IOU_THRESHOLD:.2f} = {mAP:.4f}")


    test_results = evaluate_val_set(model, test_loader)
    test_metrics = compute_metrics(test_results)
    test_ap_dict, test_mAP = compute_map(test_results, NUM_CLASSES)


    print("\nTEST RESULTS") 

    print(f"Test Precision : {test_metrics['precision']:.4f}")
    print(f"Test Recall    : {test_metrics['recall']:.4f}")
    print(f"Test F1 Score  : {test_metrics['f1']:.4f}")
    print(f"Test Accuracy  : {test_metrics['accuracy']:.4f}")
    print(f"Test mAP@50    : {test_mAP:.4f}")



    class_names = [FOOD_MAP[i] for i in range(len(FOOD_MAP))]
    print("\n  Per-class AP:")
    for cls_id, ap in sorted(test_ap_dict.items()):
        name = FOOD_MAP.get(cls_id - 1, f"cls_{cls_id}")
        print(f"    {name:20s}  AP = {ap:.4f}")

    tp_total  = sum(1 for _, is_tp, _, _ in test_results if is_tp)
    det_total = len(test_results)
    accuracy  = tp_total / det_total if det_total > 0 else 0.0
    print(f"\n  Detection Accuracy = {accuracy:.4f} ({tp_total}/{det_total})")

    
    plot_per_class_ap(test_ap_dict, class_names, save_path="per_class_ap_test.png")  

    print(f"Final mAP@50      : {mAP:.4f}")
    print(f"Detection Accuracy: {accuracy:.4f}")
    print(f"Best Val Loss     : {best_f1:.4f}")

    plot_loss_curves(train_loss_history, val_loss_history)
    plot_metric_curves(
    precision_history,
    recall_history,
    f1_history,
    accuracy_history
    )
    test_conf_mat = build_confusion_matrix(test_results, NUM_CLASSES)

    plot_confusion_matrix(
        test_conf_mat,
        class_names,
        save_path="test_confusion_matrix.png"
    )


#LOAD/PREDICT kaydedilmiş modeli yükler
def load_model(path="best_food_model.pth"):
    model = get_model(NUM_CLASSES, fine_tune=False)
    model.load_state_dict(
        torch.load(
            path, map_location=device, weights_only=True
            )
        )
    model.to(device)
    model.eval()
    print(f"Model loaded from: {path}")
    return model



import torchvision.ops as ops

def predict(model, img_path, threshold=CONF_THRESHOLD):
    img = imread(img_path)
    if img is None:
        print("Image could not be read.")
        return

    #img preprocessing
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_rgb     = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    tensor      = transforms.ToTensor()(img_rgb).to(device)


    import time  # inference decay
    with torch.no_grad():

    # GPU memory baseline
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        # CPU memory baseline (Linux/Windows)
        process = None
        try:
            import psutil
            process = psutil.Process(os.getpid())
            cpu_mem_before = process.memory_info().rss / 1024 ** 2  # MB
        except:
            cpu_mem_before = None

        #TIME START
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.time()

        output = model([tensor])[0]

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end = time.time()
        # ---- TIME END ----

        # GPU memory peak
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.max_memory_allocated() / 1024 ** 2  # MB
        else:
            gpu_mem = None

        # CPU memory after
        if process:
            cpu_mem_after = process.memory_info().rss / 1024 ** 2
        else:
            cpu_mem_after = None

    print(f"\nPERFORMANCE METRICS")
    print(f"Inference Time: {end - start:.4f} sec")

    if gpu_mem is not None:
        print(f"GPU Peak Memory: {gpu_mem:.2f} MB")

    if cpu_mem_before is not None:
        print(f"CPU Memory Before: {cpu_mem_before:.2f} MB")
        print(f"CPU Memory After : {cpu_mem_after:.2f} MB")
        print(f"CPU Delta        : {cpu_mem_after - cpu_mem_before:.2f} MB")



    boxes  = output["boxes"]
    labels = output["labels"]
    scores = output["scores"]

    #aynı objeye ait fazla detectionları temizliyor 
    keep = ops.batched_nms(boxes, scores, labels, iou_threshold=NMS_IOU)

    boxes  = boxes[keep].cpu().numpy()
    labels = labels[keep].cpu().numpy()
    scores = scores[keep].cpu().numpy()

    print(f"\nTotal predictions after NMS: {len(boxes)}")
    if len(scores):
        print(f"Score range: {scores.min():.3f} – {scores.max():.3f}")

    canvas     = img_rgb.copy()
    total_kcal = 0.0
    total_protein = 0.0
    total_carb = 0.0
    total_fat = 0.0

    legend     = []

    for box, label, score in zip(boxes, labels, scores):
        if score < threshold:
            continue

        yolo_id = int(label) - 1
        nutrition= estimate_calories(yolo_id, box, IMG_SIZE)

        food_name = nutrition["food"]
        weight_g = nutrition ["weight"]
        kcal = nutrition["kcal"]
        protein = nutrition["protein"]
        carb = nutrition["carb"]
        fat = nutrition["fat"]
        

        total_kcal += kcal
        total_protein += protein
        total_carb += carb
        total_fat += fat



        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 220, 0), 2)
        #güven skoru ve kalorie yemeğin 
        tag = f"{food_name} {score:.2f} ~{kcal:.0f}kcal"
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(canvas, (x1, max(y1-th-8, 0)), (x1+tw, max(y1, th+8)), (0,180,0), -1)
        cv2.putText(canvas, tag, (x1, max(y1-4, th+4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)
        legend.append(
            f"{food_name:15s} "
            f"{weight_g:.0f}g "
            f"{kcal:.0f}kcal "
            f"Protein: {protein:.1f} "
            f"Carb: {carb:.1f} "
            f"Fat: {fat:.1f} "

        )

    if not legend:
        print(f"\nNo detections above threshold {threshold}.")
        print("Try lowering CONF_THRESHOLD (e.g. 0.35).")

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(canvas)
    ax.axis("off")
    ax.set_title(f"Toplam Tahmini Kalori: {total_kcal:.0f} kcal", fontsize=14, fontweight="bold")
    if legend:
        ax.text(0.01, 0.01, "\n".join(legend),
                transform=ax.transAxes, fontsize=7.5,
                verticalalignment="bottom", fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.88))
    plt.tight_layout()
    plt.savefig("prediction.png", dpi=150, bbox_inches="tight")
    plt.show()

    print("\nKalori Özeti")
    for line in legend:
        print(" ", line)
    print(f"\n  TOPLAM: {total_kcal:.0f} kcal")
    print("\nMacro Summary")
    print(f"Protein : {total_protein:.1f} g ")
    print(f"Carbs : {total_carb:.1f} g")
    print(f"Fat : {total_fat:.1f} g")


#MAIN 
if __name__ == "__main__":
    print("1 - Eğit (Train)")
    print("2 - Test (Predict) — best_food_model.pth")
    print("3 - Test (Predict) — modelfoodplus3.pth")
    choice = input("Seçim: ").strip()

    if choice == "1":
        train()
    elif choice == "2":
        mdl  = load_model("best_food_model.pth")
        path = input("Görüntü yolu: ").strip()
        predict(mdl, path)
    elif choice == "3":
        mdl  = load_model("modelfoodplus3.pth")
        path = input("Görüntü yolu: ").strip()
        predict(mdl, path)
    else:
        print("Geçersiz seçim.")