from sklearn.decomposition import PCA
import tensorflow as tf
import torchvision.models as models
import torch
import numpy as np
import os


class CnnFeatureExtractor:
    def __init__(self, gpu, imagenet, model_name, output_layer, pca_dim, reshape=False):
        self.model_name = model_name
        self.gpu = gpu
        self.device = torch.device("cuda:" + str(self.gpu) if self.gpu != '-1' else "cpu")
        self.output_layer = output_layer
        self.imagenet = imagenet
        self.pca = PCA(n_components=pca_dim)
        self.reshape = reshape

        if self.model_name == 'ResNet50':
            self.model = tf.keras.applications.ResNet50()
        elif self.model_name == 'VGG19':
            self.model = tf.keras.applications.VGG19()
        elif self.model_name == 'AlexNet':
            self.model = models.alexnet(pretrained=True)
            self.model.to(self.device)
            self.model.eval()
        elif self.model_name == 'ResNet152':
            self.model = tf.keras.applications.ResNet152()
        else:
            raise NotImplemented('This feature extractor has not been added yet!')

    def classify(self, sample):
        image, filename = sample

        if self.model_name == 'AlexNet':
            output = torch.nn.functional.softmax(input=self.model(image[None, ...].to(self.device)), dim=1)
            return {'ImageID': os.path.splitext(filename)[0],
                    'ClassStr': self.imagenet[int(np.argmax(output.data.cpu().numpy()))],
                    'ClassNum': np.argmax(output.data.cpu().numpy()),
                    'Prob': np.amax(output.data.cpu().numpy())}
        else:
            output = self.model.predict(image, batch_size=1)
            return {'ImageID': os.path.splitext(filename)[0],
                    'ClassStr': self.imagenet[int(np.argmax(output))],
                    'ClassNum': np.argmax(output),
                    'Prob': np.amax(output)}

    def get_out_shape(self):
        if not self.reshape:
            s1 = torch.nn.Sequential(*list(self.model.children())[:-1])
            s2 = torch.nn.Flatten()
            s3 = torch.nn.Sequential(*list(list(self.model.children())[-1][1:-self.output_layer]))
            feature_model = torch.nn.Sequential(s1, s2, s3)
        else:
            feature_model = torch.nn.Sequential(*list(self.model.children())[0])
        output = feature_model(torch.rand((1, 3, 224, 224)).to(self.device)).data.cpu().numpy()
        output = np.squeeze(output)
        return output.shape

    def extract_feature(self, image):
        if self.model_name == 'AlexNet':
            if not self.reshape:
                s1 = torch.nn.Sequential(*list(self.model.children())[:-1])
                s2 = torch.nn.Flatten()
                s3 = torch.nn.Sequential(*list(list(self.model.children())[-1][1:-self.output_layer]))
                feature_model = torch.nn.Sequential(s1, s2, s3)
            else:
                feature_model = torch.nn.Sequential(*list(self.model.children())[0])
            feature_model.eval()
            output = np.squeeze(feature_model(
                image[None, ...].to(self.device)
            ).data.cpu().numpy())
        else:
            output = tf.keras.Model(self.model.input,
                                    self.model.get_layer(self.output_layer).output)(image, training=False)

        return output

    def pca_reduction(self, category_embedding):
        return self.pca.fit_transform(category_embedding)
