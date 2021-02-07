import copy
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

plt.ion()  # interactive mode


class GbcCnnService:
    DEVICE_TYPE = 'cuda:0'
    DATA_DIR = 'data/'
    NUM_EPOCHS = 10

    def __init__(self):
        self.data_loader = None
        self.inputs = None
        self.class_list = list()
        self.class_name_list = list()
        self.dataset_size_dict = dict()
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.device = torch.device(
            self.DEVICE_TYPE if torch.cuda.is_available() else "cpu")

    def load_data(self):
        """
        read dataset
        """
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

        image_datasets = {x: datasets.ImageFolder(
            os.path.join(image_service.DATA_DIR, x), data_transforms[x])
            for x in ['train', 'val']}

        self.data_loader = {x: DataLoader(
            image_datasets[x], batch_size=4, shuffle=True, num_workers=4)
            for x in ['train', 'val']}

        self.dataset_size_dict = {x: len(image_datasets[x]) for x in ['train', 'val']}
        self.class_name_list = image_datasets['train'].classes

        # Get a batch of training data to show
        self.inputs, self.class_list = next(iter(self.data_loader['train']))
        # out = torchvision.utils.make_grid(image_service.inputs)
        # image_service.image_show(out, title=[
        #     image_service.class_name_list[x] for x in image_service.class_list])
        # plt.pause(1.5)  # pause a bit so that plots are updated
        # plt.close()

    @staticmethod
    def image_show(inp, title=None):
        inp = inp.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
        plt.imshow(inp)
        if title is not None:
            plt.title(title)
        plt.pause(1.5)  # pause a bit so that plots are updated
        plt.close()

    def visualize_model(self, _num_images=6):
        was_training = self.model.training
        self.model.eval()
        images_so_far = 0
        fig = plt.figure()

        with torch.no_grad():
            for i, (_inputs, labels) in enumerate(self.data_loader['val']):
                _inputs = _inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(_inputs)
                _, preds = torch.max(outputs, 1)

                for j in range(_inputs.size()[0]):
                    images_so_far += 1
                    ax = plt.subplot(_num_images // 2, 2, images_so_far)
                    ax.axis('off')
                    ax.set_title('predicted: {}'.format(self.class_name_list[preds[j]]))
                    self.image_show(_inputs.cpu().data[j])

                    if images_so_far == _num_images:
                        self.model.train(mode=was_training)
                        return

            self.model.train(mode=was_training)
            plt.ioff()
            plt.show()

    def train_model(self, _num_epochs=10):
        since = time.time()

        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0

        for epoch in range(_num_epochs):
            print('Epoch {}/{}'.format(epoch, _num_epochs - 1))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()  # Set model to training mode
                else:
                    self.model.eval()  # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for _inputs, labels in self.data_loader[phase]:
                    _inputs = _inputs.to(self.device)
                    labels = labels.to(self.device)

                    # zero the parameter gradients
                    self.optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(_inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = self.criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()

                    # statistics
                    running_loss += loss.item() * _inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                if phase == 'train':
                    self.scheduler.step()

                epoch_loss = running_loss / self.dataset_size_dict[phase]
                epoch_acc = running_corrects.double() / self.dataset_size_dict[phase]

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(self.model.state_dict())

            print()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        # load best model weights
        self.model.load_state_dict(best_model_wts)

    def modeling(self):
        self.model = models.vgg16(pretrained=True)

        # add new classification layer
        self.model.classifier[-1] = nn.Linear(in_features=4096, out_features=len(self.class_list))

        # fixed pre-trained cnn weights
        for param in self.model.parameters():
            param.requires_grad = False

        self.model = self.model.to(image_service.device)

        self.criterion = nn.CrossEntropyLoss()

        self.optimizer = optim.SGD(self.model.classifier.parameters(), lr=0.001, momentum=0.9)

        # Decay LR by a factor of 0.1 every 7 epochs
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=7, gamma=0.1)


if __name__ == '__main__':
    image_service = GbcCnnService()

    image_service.load_data()

    image_service.modeling()

    image_service.train_model(_num_epochs=image_service.NUM_EPOCHS)

    image_service.visualize_model()
