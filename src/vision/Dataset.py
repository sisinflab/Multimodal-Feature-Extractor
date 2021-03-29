from PIL import Image
from config.configs import *
import tensorflow as tf
from torchvision import transforms
import numpy as np
import os


class Dataset:
    def __init__(self, dataset, model_name='VGG19', resize=(224, 224), normalize=True):
        self.directory = images_path.format(dataset)
        self.filenames = os.listdir(self.directory)
        self.filenames.sort(key=lambda x: int(x.split(".")[0]))
        self.num_samples = len(self.filenames)
        self.model_name = model_name
        self.resize = resize
        self.normalize = normalize

    def resize_and_normalize(self, sample):
        # resize
        if self.resize:
            res_sample = sample.resize(self.resize, resample=Image.BICUBIC)
        else:
            res_sample = sample

        # normalize
        if self.normalize:
            if self.model_name == 'ResNet50':
                norm_sample = tf.keras.applications.resnet.preprocess_input(np.array(res_sample))
            elif self.model_name == 'VGG19':
                norm_sample = tf.keras.applications.vgg19.preprocess_input(np.array(res_sample))
            elif self.model_name == 'AlexNet':
                transform = transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                     std=[0.229, 0.224, 0.225])
                                                ])
                norm_sample = transform(res_sample)
            elif self.model_name == 'ResNet152':
                norm_sample = tf.keras.applications.resnet.preprocess_input(np.array(res_sample))
            else:
                raise NotImplemented('This feature extractor has not been added yet!')
        else:
            return res_sample

        return norm_sample

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        sample = Image.open(self.directory + self.filenames[idx])

        if sample.mode != 'RGB':
            sample = sample.convert(mode='RGB')

        norm_sample = self.resize_and_normalize(sample)

        if self.model_name == 'AlexNet':
            return norm_sample, np.array(sample), self.filenames[idx]
        else:
            return np.expand_dims(norm_sample, axis=0), np.array(sample), self.filenames[idx]
