import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Input
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


class BranchMaskDiceHead(tf.keras.Model):
    def __init__(self, seg_model, classes, activation, **kwargs):
        super().__init__(**kwargs)
        # feature extractor
        segmentation_output, encoder_last = seg_model.model.output
        segmentation_input = Input(shape=segmentation_output.shape[1:])  # (64, 128, 256)
        encoder_input = Input(shape=encoder_last.shape[1:])  # (16, 32, 256)
        x = Activation(activation)(segmentation_input)
        x = MaxPool2D((4, 4), 4)(x)  # (64, 128, 256) -> (16, 32, 256)
        # Dice regression head
        x = Concatenate()([x, encoder_input])  # (16, 32, 256+classes)
        for i in range(3):
            x = Conv2dBnAct(256, (3, 3))(x)
        x = Conv2dBnAct(256, (3, 3), 2)(x)  # (16, 32, 256) -> (8, 16, 256)
        x = Flatten()(x)
        x = DenseDropout(1024, 0.2)(x)
        x = DenseDropout(1024, 0.5)(x)
        # mask dice predictor
        regression = Dense(classes)(x)

        self.regression_branch = Model(inputs=[segmentation_input, encoder_input], outputs=regression)

    def call(self, inputs):
        return self.regression_branch(inputs)


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
        # feature extractor
        self.seg_model = FeatureExtractor(Deeplabv3(
            weights_info,
            input_tensor,
            input_shape,
            classes,
            backbone,
            OS, alpha,
            activation,
            freeze,
        ), name='segmentation_branch')
        self.upsampling = Upsampling(input_shape[:2])
        self.seg_head = Activation(activation, name='segmentation_head')

        self.regression_branch = BranchMaskDiceHead(
            self.seg_model,
            classes,
            activation,
            name='regression_branch')
        self.regression_head = ReLU(max_value=1, name='mask_dice_head')

    def call(self, inputs):
        # segmentation branch
        segmentation, feature = self.seg_model(inputs)
        # mask dice regression branch
        regression = self.regression_branch([segmentation, feature])

        # postprocess
        # upsampling and softmax for mask
        segmentation = self.upsampling(segmentation)
        segmentation = self.seg_head(segmentation)
        # limit value between 0 and 1
        regression = self.regression_head(regression)

        return segmentation, regression

    def compile(self,
                seg_loss, seg_optimizer, seg_metrics,
                regression_loss, regression_optimizer, regression_metrics,
                **kwargs):
        super().compile(**kwargs)
        self.seg_loss = seg_loss
        self.seg_optimizer = seg_optimizer
        self.seg_metrics = seg_metrics
        self.regression_loss = regression_loss
        self.regression_optimizer = regression_optimizer
        self.regression_metrics = regression_metrics

    def train_step(self, inputs):
        input_image, gt_mask = inputs

        # segmentation branch
        with tf.GradientTape() as tape:
            # forward
            pr_mask, _ = self(input_image, training=True)
            seg_loss = self.seg_loss(gt_mask, pr_mask)
        # backward
        grads = tape.gradient(seg_loss, self.seg_model.trainable_variables)
        self.seg_optimizer.apply_gradients(zip(grads, self.seg_model.trainable_variables))

        # mask dice regression branch
        with tf.GradientTape() as tape:
            # forward
            pr_mask, pr_dice = self(input_image, training=True)
            # calculate dice between pr_mask and gt_mask as gt_dice
            gt_dice = _f_index(gt_mask, pr_mask, axis=[1, 2])  # (N,C)
            regression_loss = self.regression_loss(gt_dice, pr_dice)
        # backward
        grads = tape.gradient(regression_loss, self.regression_branch.trainable_variables)
        self.regression_optimizer.apply_gradients(zip(grads, self.regression_branch.trainable_variables))

        # calculate metrics
        seg_metrics = self.seg_metrics(gt_mask, pr_mask)
        self.add_metric(seg_metrics, name='seg_metrics')
        regression_metrics = self.regression_metrics(gt_dice, pr_dice)
        self.add_metric(regression_metrics, name='regression_metrics')

        return {'seg_loss': seg_loss, 'regression_loss': regression_loss,
                'seg_metrics': seg_metrics, 'regression_metrics': regression_metrics}
