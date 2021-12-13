import os
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import ReLU
from .Deeplabv3 import Deeplabv3
from ..losses.functional import _f_index


class Upsampling(tf.keras.layers.Layer):
    def __init__(self, size, **kwargs):
        super().__init__(**kwargs)
        self.size = size  # (Height, Width)

    def call(self, inputs):
        return tf.compat.v1.image.resize(
            inputs,
            self.size,
            method='bilinear',
            align_corners=True)


class FeatureExtractor(tf.keras.Model):
    def __init__(self, base_model, **kwargs):
        super().__init__(**kwargs)
        # feature extractor
        encoder_last = base_model.get_layer(name='dropout_encoder_last').output
        # mask predictor
        segmentation_output = base_model.get_layer(name='custom_logits_semantic').output
        self.model = Model(inputs=base_model.input, outputs=[segmentation_output, encoder_last])

    def call(self, inputs):
        return self.model(inputs)


class Conv2dBnAct(tf.keras.layers.Layer):
    def __init__(self, output_channels, kernel_size, stride=1, padding='same', **kwargs):
        super().__init__(**kwargs)
        self.conv = Conv2D(output_channels, kernel_size, stride, padding=padding, use_bias=False)
        self.bn = BatchNormalization(epsilon=1e-5)
        self.relu = Activation('relu')

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.bn(x)
        x = self.relu(x)
        return x


class DenseDropout(tf.keras.layers.Layer):
    def __init__(self, output_channels, drop_rate, **kwargs):
        super().__init__(**kwargs)
        self.dense = Dense(output_channels, activation='relu')
        self.dropout = Dropout(drop_rate)

    def call(self, inputs):
        x = self.dense(inputs)
        x = self.dropout(x)
        return x


class Deeplabv3WithDiceHead(tf.keras.Model):
    def __init__(
        self,
        weights_info={'weights': 'pascal_voc'},
        input_tensor=None,
        input_shape=(512, 512, 3),
        classes=21,
        backbone='xception',
        OS=16, alpha=1.,
        activation='softmax',
        freeze=False,
    ):
        super().__init__(name='deeplabv3plusWithDiceHead')
        self.base_model = FeatureExtractor(Deeplabv3(
            weights_info,
            input_tensor,
            input_shape,
            classes,
            backbone,
            OS, alpha,
            activation,
            freeze,
        ))
        self.activation = Activation(activation)
        self.pool = MaxPool2D((4, 4), 4)
        self.concat = Concatenate()
        self.conv1 = Conv2dBnAct(256, (3, 3))  # input channel: 256+classes
        self.conv2 = Conv2dBnAct(256, (3, 3))  # input channel: 256
        self.conv3 = Conv2dBnAct(256, (3, 3), 2)  # input channel: 256
        self.flatten = Flatten()
        self.dense1 = DenseDropout(1024, 0.2)  # input channel: img_height//(8*4) * img_width//(8*4) * 256
        self.dense2 = DenseDropout(1024, 0.5)  # input channel: 1024
        self.classifier = Dense(classes, name='last_dense')  # input channel: 1024
        self.dice_head = ReLU(max_value=1, name='mask_dice_head')
        self.upsampling = Upsampling(input_shape[:2])
        self.seg_head = Activation(activation, name='segmentation_head')

    def call(self, inputs):
        # feature extractor
        segmentation, encoder_last = self.base_model(inputs)
        # Mask Dice Head
        x = self.pool(self.activation(segmentation))
        x = self.concat([x, encoder_last])
        x = self.conv1(x)
        for _ in range(2):
            x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.classifier(x)
        regression = self.dice_head(x)

        # Segmentation Head
        segmentation = self.upsampling(segmentation)
        segmentation = self.seg_head(segmentation)

        return segmentation, regression

    def compile(self, seg_loss, seg_optimizer, regression_loss, regression_optimizer):
        super().compile()
        self.seg_loss = seg_loss
        self.seg_optimizer = seg_optimizer
        self.regression_loss = regression_loss
        self.regression_optimizer = regression_optimizer

    def train_step(self, inputs):
        input_image, gt_mask = inputs

        with tf.GradientTape() as tape:
            # forward
            pr_mask, pr_dice = self(input_image, training=True)
            # calculate dice between pr_mask and gt_mask as gt_dice
            gt_dice = _f_index(gt_mask, pr_mask, axis=[1, 2])  # (N,C)
            # compute loss
            seg_loss = self.seg_loss(gt_mask, pr_mask)
            regression_loss = self.regression_loss(gt_dice, pr_dice)
            loss = seg_loss + regression_loss

        grads = tape.gradient(loss, self.trainable_variables)
        self.seg_optimizer.apply_gradients(zip(grads, self.trainable_variables))

        return {'seg_loss': seg_loss, 'regression_loss': regression_loss}
