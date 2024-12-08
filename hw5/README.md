## AIoT Applications and Data Analysis

Course ID: 7716, CS, NCHU  
Dec., 2024

### HW5 - Deep Lerarning Basic
![](docs/imgs/hw5_demo1.jpg)<br>Iris | ![](docs/imgs/hw5_demo2.jpg)<br>MNIST | ![](docs/imgs/hw5_demo3.jpg)<br>CARFI10
|:-:|:-:|:-:|


#### Library Dependencies
Install Python libraries
```bash
python -m pip install scikit-learn==1.2.2 tensorflow==2.13.1 pytorch-lightning==2.4.0 torchvision==0.17.2
```

#### Running
```bash
python main.py && python main2.py && python main3.py
```

#### Logging
The log files in `doscs/`.
```
tensorboard --logdir_spec=keras:logs/1/keras/validation,torch:logs/1/torch,lightning:logs/1/pl
```

```
tensorboard --logdir_spec=dense:logs/2/dense/validation,conv:logs/2/conv/validation
```

```
tensorboard --logdir_spec=vgg16:logs/3/vgg16/version_0,vgg19:logs/3/vgg19/version_0
```


#### Exercise
##### 5.1 - Iris classification problem in tf.keras, PyTorch and PyTorch Lightning
##### :speech_balloon: Step 1
###### :bulb: *Prompt*
solve iris classfication problem with early stopping, batch normalzation and dropout in keras, pytorch and pytorch lightnning, please.

###### ![](docs/imgs/icons8-chatgpt.svg) *ChatGPT:*
Here is a complete implementation of the Iris classification problem using Keras, PyTorch, and PyTorch Lightning, incorporating early stopping, batch normalization, and dropout.

```python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
import tensorflow as tf
import keras
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import logging
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
import warnings
warnings.filterwarnings("ignore", ".*does not have many workers.*")

import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from keras.callbacks import TensorBoard
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

def load_iris():
    iris = datasets.load_iris();
    x = iris.data
    y = iris.target
    
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.17,random_state=42)
    x_train, x_val, y_train, y_val = train_test_split(x_train,y_train,test_size=0.17,random_state=42)
    
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)
    x_test = scaler.transform(x_test)
    return x_train, x_val, x_test, y_train, y_val, y_test

x_train, x_val, x_test, y_train, y_val, y_test = load_iris()

batch_size = 8
max_epochs = 20

def _keras():
    print("Keras:")
    num_classes = 3
    _y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    _y_val = tf.keras.utils.to_categorical(y_val, num_classes)
    _y_test = tf.keras.utils.to_categorical(y_test, num_classes)
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(x_train.shape[1],)),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    early_stopping = keras.callbacks.EarlyStopping(monitor='loss', patience=2, restore_best_weights=True, verbose=1)
    tbcb = TensorBoard(log_dir="logs/1/keras", histogram_freq=0)
    model.fit(x_train, _y_train, epochs=max_epochs, batch_size=batch_size,  validation_data=(x_val, _y_val), callbacks=[early_stopping, tbcb], verbose=0)
    test_loss, test_acc = model.evaluate(x_test, _y_test, verbose=1)
    print(f"Test accuracy: {test_acc}\n")

_keras()

x_train, x_val, x_test, y_train, y_val, y_test = load_iris()
x_train = torch.tensor(x_train, dtype=torch.float32)
x_val = torch.tensor(x_val, dtype=torch.float32)
x_test = torch.tensor(x_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_val = torch.tensor(y_val, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

batch_size = 8
train_dataset = TensorDataset(x_train, y_train)
val_dataset = TensorDataset(x_val, y_val)
test_dataset = TensorDataset(x_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

def _torch():
    print("Torch:")

    class IrisNet(nn.Module):
        def __init__(self):
            super(IrisNet, self).__init__()
            self.fc1 = nn.Linear(4, 16)
            self.bn1 = nn.BatchNorm1d(16)
            self.dropout1 = nn.Dropout(0.3)
            self.fc2 = nn.Linear(16, 8)
            self.bn2 = nn.BatchNorm1d(8)
            self.dropout2 = nn.Dropout(0.3)
            self.fc3 = nn.Linear(8, 3)
        
        def forward(self, x):
            x = self.dropout1(torch.relu(self.bn1(self.fc1(x))))
            x = self.dropout2(torch.relu(self.bn2(self.fc2(x))))
            x = self.fc3(x)
            return x

    model = IrisNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    writer = SummaryWriter(log_dir='logs/1/torch')

    # Early Stopping parameters
    patience = 2  
    best_val_loss = float('inf')  
    epochs_without_improvement = 0  
    best_model_state = None

    epochs = max_epochs
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for feature, labels in train_loader:
            outputs = model(feature)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for feature, labels in val_loader:
                outputs = model(feature)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_accuracy = correct / total
        writer.add_scalar('epoch_loss', val_loss / len(val_loader), epoch)
        writer.add_scalar('epoch_accuracy', val_accuracy, epoch)

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            # Save the best model
            best_model_state = model.state_dict()
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for features, labels in test_loader:
            outputs = model(features)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.4f}\n")

_torch()


def _pl():
    print("Lightning:")
    writer = SummaryWriter(log_dir='logs/2/pl')
    class IrisClassifier(pl.LightningModule):

        def __init__(self, input_size=4, 
                    hidden_size1=16,
                    hidden_size2=8,
                    output_size=3,
                    lr=0.01):
            super(IrisClassifier, self).__init__()
            self.model = nn.Sequential(
                nn.Linear(input_size, hidden_size1),
                nn.BatchNorm1d(hidden_size1),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_size1, hidden_size2),
                nn.BatchNorm1d(hidden_size2),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_size2, output_size)
            )
            self.criterion = nn.CrossEntropyLoss()
            self.lr = lr
            self.test_acc = []
            self.val_acc = []
            self.val_loss = []

        def forward(self, x):
            return self.model(x)
        
        def training_step(self, batch, batch_idx):
            x, y = batch
            y_hat = self(x)
            loss = self.criterion(y_hat, y)
            # self.log("loss/train", loss)
            return loss
        
        @torch.no_grad()
        def validation_step(self, batch, batch_idx):
            loss, acc = self.validate(batch)
            self.val_acc += [acc.cpu().numpy()]
            self.val_loss += [loss.cpu().numpy()]
            self.log("val_loss", loss, prog_bar=False)
            self.log("val_accuracy", acc, prog_bar=False)
        
        def validate(self, batch):
            x, y = batch
            y_hat = self(x)
            loss = self.criterion(y_hat, y)
            acc = (y_hat.argmax(dim=1)==y).float().mean()
            return loss, acc

        @torch.no_grad()
        def test_step(self, batch, batch_idx):
            loss, acc = self.validate(batch)
            self.test_acc += [acc.cpu().numpy()]
            # self.log("test_loss", loss, prog_bar=False)
            # self.log("test_acc", acc, prog_bar=False)
        
        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters(), lr=self.lr)

        def evaluate(self):
            return np.array(self.test_acc).mean()
        
        def on_validation_epoch_end(self):
            writer.add_scalar('epoch_loss', np.array(self.val_loss).mean(), self.current_epoch)
            writer.add_scalar('epoch_accuracy', np.array(self.val_acc).mean(), self.current_epoch)
            self.val_loss.clear()
            self.val_acc.clear()

    model = IrisClassifier()
    early_stopping = pl.callbacks.EarlyStopping(monitor="val_loss", patience=2, verbose=True, mode="min")
    trainer = pl.Trainer(max_epochs=20, log_every_n_steps=1, callbacks=[early_stopping], enable_progress_bar=False)
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    trainer.test(model, dataloaders=test_loader, verbose=False)
    acc = model.evaluate()
    print(f"Test Accuracy: {acc:.4f}")

_pl()
```

This implementation follows the same design principles in all three frameworks, ensuring comparability. Each method incorporates early stopping, batch normalization, and dropout.

----

##### 5.2 - MNIST classification problem in dense and convolutional
##### :speech_balloon: Step 1
###### :bulb: *Prompt*
solve mnist classification problem with early stopping, batch normalzation and dropout dense and convolutional, please.

###### ![](docs/imgs/icons8-chatgpt.svg) *ChatGPT:*
Here’s the Python script:
```python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

import tensorflow as tf
from sklearn.model_selection import train_test_split
from tf.keras.datasets import mnist
from tf.keras.utils import to_categorical
from tf.keras.models import Sequential
from tf.keras.layers import Flatten, Dense, BatchNormalization, Dropout, Conv2D, MaxPooling2D
from keras.callbacks import TensorBoard, EarlyStopping

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_val, y_train, y_val = train_test_split(x_train,y_train,test_size=0.17,random_state=42)

x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
x_val = x_val.reshape(-1, 28, 28, 1).astype('float32') / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
y_train = to_categorical(y_train, 10)
y_val = to_categorical(y_val, 10)
y_test = to_categorical(y_test, 10)

batch_size = 32
max_epochs = 2

def _fit(model, log_dir):
    early_stopping = EarlyStopping(monitor='loss', patience=2, restore_best_weights=True, verbose=1)
    tbcb = TensorBoard(log_dir=f"2_logs/{log_dir}", histogram_freq=0)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'] )
    model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=max_epochs, batch_size=batch_size, callbacks=[early_stopping, tbcb])
    loss, accuracy = model.evaluate(x_test, y_test)
    print(f"Test Accuracy: {accuracy:.4f}")

def _dense():
    model = Sequential([
        tf.keras.Input(shape=(28, 28, 1)),
        Flatten(),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(10, activation='softmax')
    ])
    return model

def _conv():
    model = Sequential([
        tf.keras.Input(shape=(28, 28, 1)),
        Conv2D(32, kernel_size=(3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.2),
        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.2),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])
    return model

_fit(_dense(), "dense")
_fit(_conv(), "conv")
```

**Key Features:**

**1. Data Preparation:**
  - Normalization and reshaping for compatibility with Keras layers.
  - One-hot encoding for classification labels.

**2. Model Architectures:**
  - Dense model with fully connected layers, dropout, and batch normalization.
  - Convolutional model with Conv2D, max pooling, dropout, and batch normalization.

**3. Callbacks:**
  - Early stopping prevents overfitting by halting training when the loss does not improve.
  - TensorBoard logging allows visualization of training metrics.

**4. Training and Evaluation:**
  - Both models are trained and evaluated separately, with test accuracy printed for each.


  ----

##### 5.3 - CIFAR10 in VGG16 and VGG19
##### :speech_balloon: Step 1
###### :bulb: *Prompt*
solve cifar10 classification problem in pretrained models of vgg16 and vgg19 in pytorch lightning, please.

###### ![](docs/imgs/icons8-chatgpt.svg) *ChatGPT:*
Here's a Python script implementing your requirements:
```python
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
```

This script fulfills all the requirements specified. It uses PyTorch Lightning for training and evaluation, employs pre-trained VGG models, and properly handles dataset loading, logging, and evaluation.