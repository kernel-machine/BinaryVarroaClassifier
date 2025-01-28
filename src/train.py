import torch
import torchvision
import argparse
import numpy as np
from datetime import datetime
import os
import json
import torchvision.transforms.v2
from lib.varroa_frame_dataset import VarroaDataset
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=32, type=int)
parser.add_argument("--lr", default=1e-3, type=float)
parser.add_argument("--epochs", default=50, type=int)
parser.add_argument("--input_dataset", required=True, type=str)
parser.add_argument("--evaluate", default=False, action="store_true")

args = parser.parse_args()
SEED = 1234
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
now = datetime.now()  # current date and time
date_time = now.strftime("%Y-%b-%d-%H:%M:%S")
log_dir = os.path.join("../runs", date_time)
writer = SummaryWriter(log_dir=log_dir)
with open(os.path.join(log_dir,"args.json"),"w") as f:
    f.write(json.dumps(vars(args)))
    f.close()

image_loader = torchvision.transforms.v2.Compose(
    [        
        torchvision.transforms.v2.Resize((256,256)),
        torchvision.transforms.v2.ToDtype(torch.float32),
        torchvision.transforms.v2.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ]
)

image_loader_aug = torchvision.transforms.v2.Compose(
    [
        torchvision.transforms.v2.Resize((256,256)),
        torchvision.transforms.v2.AugMix(),
        torchvision.transforms.v2.ToDtype(torch.float32),
        torchvision.transforms.v2.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ]
)

train_set = VarroaDataset(os.path.join(args.input_dataset,"train"), image_processing=image_loader_aug)
val_set = VarroaDataset(os.path.join(args.input_dataset,"val"), image_processing=image_loader, balance=True)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=6)
test_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size, shuffle=True, num_workers=4)

model = torchvision.models.resnet18(
    weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1
)
model.fc = torch.nn.Linear(in_features=model.fc.in_features, out_features=1, bias=True)
device = torch.device("cuda:0")
model.to(device)
pos_weight = torch.tensor([train_set.varroa_free_count()/train_set.varroa_infested_count()]).to(device)
loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

running_loss = 0
correct_predictions = 0
total_predictions = 0

best_validation_loss = float("inf")

for epoch in range(args.epochs):
    for i, (imgs, labels) in enumerate(train_loader):
        model.train()
        imgs = imgs.to(device)
        labels = labels.to(device).float()
        optimizer.zero_grad()
        logits = model(imgs).flatten()
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        predicted_classes = torch.sigmoid(logits).round()
        correct_predictions += (predicted_classes == labels).sum().item()
        total_predictions += labels.size(0)
        running_loss += loss.item()

        current_loss = running_loss / (i + 1)
        current_accuracy = correct_predictions / total_predictions

        print(f"Train: Epoch {epoch} | Batch {i+1}/{len(train_loader)}, Loss: {current_loss:.4f}, Acc: {current_accuracy:.4f}", end="\r" if i+1< len(train_loader) else "\n")

    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = correct_predictions / total_predictions
    writer.add_scalar("Loss/train", epoch_loss, epoch)
    writer.add_scalar("Acc/train", epoch_accuracy, epoch)
    writer.flush()

    correct_predictions = 0
    total_predictions = 0
    running_loss = 0
    with torch.no_grad():
        for i, (imgs, labels) in enumerate(test_loader):
            model.eval()
            imgs = imgs.to(device)
            labels = labels.to(device).float()
            logits = model(imgs).flatten()
            loss = loss_fn(logits, labels)

            predicted_classes = torch.sigmoid(logits).round()
            correct_predictions += (predicted_classes == labels).sum().item()
            total_predictions += labels.size(0)
            running_loss += loss.item()

            current_loss = running_loss / (i + 1)
            current_accuracy = correct_predictions / total_predictions

            print(f"Val: Epoch {epoch} | Batch {i+1}/{len(test_loader)}, Loss: {current_loss:.4f}, Acc: {current_accuracy:.4f}", end="\r" if i+1< len(test_loader) else "\n")
    
    epoch_loss = running_loss / len(test_loader)
    epoch_accuracy = correct_predictions / total_predictions
    writer.add_scalar("Loss/val", epoch_loss, epoch)
    writer.add_scalar("Acc/val", epoch_accuracy, epoch)
    writer.flush()

    if epoch_loss < best_validation_loss:
        best_validation_loss = epoch_loss
        torch.save(model.state_dict(), os.path.join(log_dir, "model.pth"))
        print(f"Best loss update in epoch {epoch}")

