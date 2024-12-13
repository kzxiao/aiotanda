import torch
import torch.nn as nn
import pytorch_lightning as pl
import logging
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Subset
from torchvision import datasets, transforms
from torchvision.models import vgg16, vgg19
from torchsummary import summary
import warnings
warnings.filterwarnings("ignore", ".*Palette images with Transparency.*")

transform = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor(),
])

train_transform = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30),
    transform
])

dataset_path = "Face-Mask-Detection/dataset"
train_dataset = datasets.ImageFolder(dataset_path, transform=train_transform)
dataset = datasets.ImageFolder(dataset_path, transform=transform)

def load_data():
    # use different transform, instead of train, val, test = torch.utils.data.random_split(dataset, [l, r, r]).
    l=len(dataset) 
    indices = list(range(l))
    r = (int)(l*0.17)
    l -= r<<1
    r += l
    
    np.random.shuffle(indices)
    train_idx, val_idx, test_idx = indices[:l], indices[l:r], indices[r:]
    # print(f'[{l}], [{l}:{r}], [{r}:]')

    train = Subset(train_dataset, indices=train_idx)
    val = Subset(dataset, indices=val_idx)
    test = Subset(dataset, indices=test_idx)

    batch_size = 64
    num_workers = 2
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val, batch_size=batch_size, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, num_workers=num_workers)
    return train_loader, val_loader, test_loader

train_loader, val_loader, test_loader = load_data()
print(f'train: {len(train_loader.dataset)}, valid: {len(val_loader.dataset)}, test: {len(test_loader.dataset)}')

def load_model(name):
    model, l = (vgg19(weights=torchvision.models.VGG19_Weights.IMAGENET1K_V1), -9) if name == 'vgg19' else (vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1), -7)
    for p in model.features[:l]: p.require_grad = False
    in_features = model.classifier[0].out_features
    model.classifier[-4] = nn.Linear(in_features, 512)
    model.classifier[-1] = nn.Linear(512, 2)    
    # for i, m in enumerate(model.children()): 
    #     if i==2: print(f'{i}. {m}')
    return model


class FaceMaskClassifier(pl.LightningModule):
    def __init__(self, model_name="vgg19", num_classes=2, lr=1e-3):
        super().__init__()
        self.lr = lr
        self.model = load_model(model_name)
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
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
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
        optimizer = torch.optim.SGD(self.parameters(), lr=0.01, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode="min",factor=0.1,patience=3,verbose=True)
        return {"optimizer": optimizer,"lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss", "frequency": 1, "strict": True}}
    
    def evaluate(self):
        return np.array(self.test_acc).mean()
        
def _fit(model_name):
    print(f"{model_name}:")
    max_epochs = 20
    mcp = pl.callbacks.ModelCheckpoint(monitor="val_loss", dirpath=f"checkpoints/{model_name}", save_top_k=1, mode="min")
    es = pl.callbacks.EarlyStopping(monitor="val_loss", patience=2, verbose=True, mode="min")
    rpb = pl.callbacks.RichProgressBar()
    tb = pl.loggers.TensorBoardLogger(save_dir=f"logs/1/{model_name}", name=None)
    model = FaceMaskClassifier(model_name=model_name, num_classes=2, lr=1e-3)
    trainer = pl.Trainer(max_epochs=max_epochs,log_every_n_steps=10, logger=tb, callbacks=[rpb, es, mcp])
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, dataloaders=test_loader)
    acc = model.evaluate()
    print(f"Test Accuracy: {acc:.4f}")
    return model

def main():
    _fit('vgg19')
    _fit('vgg16')

if __name__ == '__main__':
    main()