import os
import torch
import torchvision
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from torch.utils.data.dataloader import DataLoader
import torchvision.transforms as tt
from torch.utils.data import random_split
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
matplotlib.rcParams['figure.facecolor'] = '#ffffff'





# Establecer el directorio de datos
data_dir = './data_center_reduced_4'  # Cambia esta ruta a donde esté tu carpeta all_data

# Comprobar las clases disponibles
classes = os.listdir(data_dir)
#print("Clases:", classes)

# Definir estadísticas para normalización (imágenes de un canal)
stats = ((0.5,), (0.5,))  # Ajusta estos valores si tienes estadísticas específicas

# Definir las transformaciones para entrenamiento y validación
train_tfms = tt.Compose([
    tt.Grayscale(num_output_channels=1),
    # Puedes agregar más transformaciones de aumento de datos aquí
    tt.RandomHorizontalFlip(),
    tt.ToTensor(),
    tt.Normalize(*stats)
])

valid_tfms = tt.Compose([
    tt.Grayscale(num_output_channels=1),
    tt.ToTensor(),
    tt.Normalize(*stats)
])

# Crear el dataset completo
dataset = ImageFolder(data_dir, transform=train_tfms)

# Dividir el dataset en entrenamiento y validación
torch.manual_seed(42)  # Para reproducibilidad
val_size = int(len(dataset) * 0.2)  # 20% para validación
train_size = len(dataset) - val_size

train_ds, valid_ds = random_split(dataset, [train_size, val_size])

# Aplicar transformaciones específicas a validación
from torch.utils.data import Subset

class SubsetWithTransform(Subset):
    def __init__(self, subset, transform=None):
        super().__init__(subset.dataset, subset.indices)
        self.transform = transform

    def __getitem__(self, index):
        x, y = super().__getitem__(index)
        if self.transform:
            x = self.transform(x)
        return x, y

# Crear datasets con transformaciones adecuadas
train_ds = SubsetWithTransform(train_ds, transform=train_tfms)
valid_ds = SubsetWithTransform(valid_ds, transform=valid_tfms)

# Definir el DataLoader personalizado para mover datos al dispositivo
class DeviceDataLoader():
    def __init__(self, dl):
        self.dl = dl
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    
        # self.device = torch.device('cpu')

    def __iter__(self):
        for b in self.dl:
            yield self.to_device(b)

    def __len__(self):
        return len(self.dl)

    def to_device(self, data):
        if isinstance(data, (list, tuple)):
            return [self.to_device(x) for x in data]
        return data.to(self.device, non_blocking=True)

# Establecer el tamaño del batch (ajusta según tu memoria)
batch_size = 8

# Crear los DataLoaders
train_dl = DataLoader(train_ds, batch_size, shuffle=False, num_workers=2, pin_memory=True)
valid_dl = DataLoader(valid_ds, batch_size*2, num_workers=2, pin_memory=True)

train_dl = DeviceDataLoader(train_dl)
valid_dl = DeviceDataLoader(valid_dl)

# Definir la clase base para el modelo
class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        out = self(images)                  # Generar predicciones
        loss = F.cross_entropy(out, labels) # Calcular pérdida
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)                  # Generar predicciones
        loss = F.cross_entropy(out, labels) # Calcular pérdida
        acc = accuracy(out, labels)         # Calcular precisión
        return {'val_loss': loss.detach(), 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combinar pérdidas
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combinar precisiones
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print(f"Epoch [{epoch}], train_loss: {result['train_loss']:.4f}, "
              f"val_loss: {result['val_loss']:.4f}, val_acc: {result['val_acc']:.4f}")

# Función para calcular la precisión
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

# Definir el bloque de convolución
def conv_block(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
              nn.BatchNorm2d(out_channels),
              nn.ReLU(inplace=True)]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)

# Definir el modelo ResNet9
class ResNet9(ImageClassificationBase):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.conv1 = conv_block(in_channels, 64)
        self.conv2 = conv_block(64, 128, pool=True)
        self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))
        self.conv3 = conv_block(128, 256, pool=True)
        self.conv4 = conv_block(256, 512, pool=True)
        self.res2 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )

    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out




# Funciones para evaluación y entrenamiento
@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def fit_one_cycle(epochs, max_lr, model, train_loader, val_loader,
                  weight_decay=0, grad_clip=None, opt_func=torch.optim.SGD):
    torch.cuda.empty_cache()
    history = []
    optimizer = opt_func(model.parameters(), max_lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr, epochs=epochs, steps_per_epoch=len(train_loader))
    for epoch in range(epochs):
        model.train()
        train_losses = []
        lrs = []
        for batch in train_loader:
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            if grad_clip:
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)
            optimizer.step()
            optimizer.zero_grad()
            lrs.append(get_lr(optimizer))
            sched.step()
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        result['lrs'] = lrs
        model.epoch_end(epoch, result)
        history.append(result)
    return history

def plot_accuracies(history):
    accuracies = [x['val_acc'] for x in history]
    plt.plot(accuracies, '-x')
    plt.xlabel('Epoch')
    plt.ylabel('Precisión')
    plt.title('Precisión de Validación')

def plot_losses(history):
    train_losses = [x.get('train_loss') for x in history]
    val_losses = [x['val_loss'] for x in history]
    plt.plot(train_losses, '-bx')
    plt.plot(val_losses, '-rx')
    plt.xlabel('Epoch')
    plt.ylabel('Pérdida')
    plt.legend(['Entrenamiento', 'Validación'])
    plt.title('Pérdida durante el Entrenamiento')

def plot_lrs(history):
    lrs = np.concatenate([x.get('lrs', []) for x in history])
    plt.plot(lrs)
    plt.xlabel('Número de Batch')
    plt.ylabel('Learning Rate')
    plt.title('Tasa de Aprendizaje durante el Entrenamiento')

if __name__ == '__main__':
    # Configuración de entrenamiento
    epochs = 20
    max_lr = 0.01
    grad_clip = 0.1
    weight_decay = 1e-4
    opt_func = torch.optim.Adam

    # Instanciar el modelo
    model = ResNet9(in_channels=1, num_classes=12)
    model = model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    # Entrenar el modelo
    history = fit_one_cycle(epochs, max_lr, model, train_dl, valid_dl,
                            grad_clip=grad_clip,
                            weight_decay=weight_decay,
                            opt_func=opt_func)

    # Especifica la ruta donde quieres guardar el modelo
    model_path = './save_model/model.pth'
# Crea el directorio si no existe
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    # Guarda el estado del modelo
    torch.save(model.state_dict(), model_path)

    # Funciones para graficar (opcional)

    # Mostrar las gráficas (opcional)
    plot_accuracies(history)
    plt.show()
    plot_losses(history)
    plt.show()
    plot_lrs(history)
    plt.show()
