import torch
import torchvision
import argparse
import numpy as np
from datetime import datetime
import os
import json
import torchvision.transforms.v2
from lib.varroa_frame_dataset import VarroaDataset
from lib.validation_metrics import ValidationMetrics

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=32, type=int)
parser.add_argument("--input_dataset", required=True, type=str)
parser.add_argument("--model", type=str, required=True)

args = parser.parse_args()
SEED = 1234
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

image_loader = torchvision.transforms.v2.Compose(
    [        
        torchvision.transforms.v2.Resize((256,256)),
        torchvision.transforms.v2.ToDtype(torch.float32),
        torchvision.transforms.v2.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ]
)
val_set = VarroaDataset(args.input_dataset, image_processing=image_loader, balance=True)
test_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size, shuffle=True, num_workers=4)

model = torchvision.models.resnet18(
    weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1
)
model.fc = torch.nn.Linear(in_features=model.fc.in_features, out_features=1, bias=True)
device = torch.device("cuda:0")
model.to(device)
model.load_state_dict(torch.load(args.model, weights_only=True))

vm = ValidationMetrics()
correct_predictions = 0
total_predictions = 0
model.eval()
with torch.no_grad():
    for i, (imgs, labels) in enumerate(test_loader):
        imgs = imgs.to(device)
        labels = labels.to(device).float()
        logits = model(imgs).flatten()

        predicted_classes = torch.sigmoid(logits).round()
        correct_predictions += (predicted_classes == labels).sum().item()
        total_predictions += labels.size(0)

        for j in range(len(predicted_classes)):
            prediction = predicted_classes[j].cpu()>0.5
            vm.add_prediction(bool(prediction), bool(labels[j]))

        current_accuracy = correct_predictions / total_predictions

        print(f"Val: Batch {i+1}/{len(test_loader)}, Acc: {current_accuracy:.4f}", end="\r" if i+1< len(test_loader) else "\n")

print(f"F1: {vm.get_f1()}")
tp, fp, tn, fn = vm.get_metrics()
print(f"TP: {tp}, FP: {fp}, TN: {tn}, FN:{fn}")
cm = vm.get_confusion_matrix()
dir_path = os.path.dirname(args.model)
cm.figure_.savefig(os.path.join(dir_path,"confusion_matrix.png"))

