import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
import logging
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
import warnings
warnings.filterwarnings("ignore", ".*'pretrained' is deprecated.*")
import torchvision

from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.models import vgg16, vgg19
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from pathlib import Path

def load_cifar10():
    # https://pytorch.org/vision/main/models/generated/torchvision.models.vgg16.html#torchvision.models.VGG16_Weights
    # https://pytorch.org/vision/main/models/generated/torchvision.models.vgg16.html#torchvision.models.VGG16_Weights
    # transform = transforms.Compose([
    #     transforms.Resize((224, 224)),
    #     # transforms.CenterCrop((224, 224)),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # ])
    transform = transforms.Compose([torchvision.models.VGG16_Weights.IMAGENET1K_FEATURES.transforms()])
    root = "data"
    download = not (Path(root) / CIFAR10.base_folder).exists()
    train_dataset = CIFAR10(root=root, train=True, download=download, transform=transform)
    test_dataset = CIFAR10(root=root, train=False, download=download, transform=transform)
    
    n_train = len(train_dataset)
    indices = list(range(n_train))
    np.random.shuffle(indices)
    split = int(np.floor(0.17 * n_train))
    train_idx, valid_idx = indices[split:], indices[:split]

    batch_size = 32
    num_workers = 2
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=SubsetRandomSampler(train_idx), num_workers=num_workers)
    val_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=SubsetRandomSampler(valid_idx), num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader

train_loader, val_loader, test_loader = load_cifar10()
print(f"train: {len(train_loader.sampler)}, val: {len(val_loader.sampler)}, test: {len(test_loader.sampler)}")

max_epochs = 2

class CifarClassifier(pl.LightningModule):
    def __init__(self, model_name="vgg16", num_classes=10, lr=1e-3):
        super().__init__()
        self.lr = lr
        self.model = vgg19(weights=torchvision.models.VGG19_Weights.IMAGENET1K_V1) if model_name == "vgg19" else vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1)
        for p in self.model.features.parameters():
            p.require_grad = False
        
        in_features = self.model.classifier[-1].in_features
        self.model.classifier[-1] = nn.Linear(in_features, num_classes)

        self.criterion = nn.CrossEntropyLoss()
        self.test_acc = []
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, acc = self.validate(batch)
        self.log("val_loss", loss, prog_bar=False)
        self.log("val_acc", acc, prog_bar=False)
        return loss

    @torch.no_grad()
    def validate(self, batch):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        acc = (y_hat.argmax(dim=1)==y).float().mean()
        return loss, acc

    def test_step(self, batch, batch_idx):
        loss, acc = self.validate(batch)
        self.test_acc += [acc.cpu().numpy()]
        return loss, acc
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
    
    def evaluate(self):
        return np.array(self.test_acc).mean()

def _fit(model_name):
    print(f"{model_name}:")
    tb = pl.loggers.TensorBoardLogger(save_dir=f"logs/3/{model_name}", name=None)
    model = CifarClassifier(model_name=model_name, num_classes=10, lr=1e-3)
    trainer = pl.Trainer(max_epochs=max_epochs,log_every_n_steps=10, logger=tb)
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, dataloaders=test_loader)
    acc = model.evaluate()
    print(f"Test Accuracy: {acc:.4f}")

def main():
    _fit("vgg16")
    _fit("vgg19")

if __name__ == '__main__':
    main()