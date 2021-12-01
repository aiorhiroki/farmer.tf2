# -*- coding: utf-8 -*-

""" Deeplabv3+ model for Keras.
This model is based on TF repo:
https://github.com/tensorflow/models/tree/master/research/deeplab
On Pascal VOC, original model gets to 84.56% mIOU

MobileNetv2 backbone is based on this repo:
https://github.com/JonathanCMitchell/mobilenet_v2_keras

# Reference
- [Encoder-Decoder with Atrous Separable Convolution
    for Semantic Image Segmentation](https://arxiv.org/pdf/1802.02611.pdf)
- [Xception: Deep Learning with Depthwise Separable Convolutions]
    (https://arxiv.org/abs/1610.02357)
- [Inverted Residuals and Linear Bottlenecks: Mobile Networks for
    Classification, Detection and Segmentation](https://arxiv.org/abs/1801.04381)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow

from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.layers import Lambda
from tensorflow.python.keras.layers import Activation
from tensorflow.python.keras.layers import Concatenate
from tensorflow.python.keras.layers import Dropout
from tensorflow.python.keras.layers import BatchNormalization
from tensorflow.python.keras.layers import Conv2D
from tensorflow.python.keras.layers import GlobalAveragePooling2D
from tensorflow.python.keras.layers import MaxPool2D
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Flatten
from tensorflow.python.keras.layers import ReLU
from tensorflow.python.keras.utils.layer_utils import get_source_inputs
from tensorflow.python.keras.utils.data_utils import get_file
from tensorflow.python.keras.applications.imagenet_utils import preprocess_input

from tensorflow.keras.applications import efficientnet
from .functional import SepConv_BN
from .dilated_xception import DilatedXception
from .mobilenetv2 import MobileNetV2

WEIGHTS_PATH_X = "https://github.com/bonlime/keras-deeplab-v3-plus/releases/download/1.1/deeplabv3_xception_tf_dim_ordering_tf_kernels.h5"
WEIGHTS_PATH_MOBILE = "https://github.com/bonlime/keras-deeplab-v3-plus/releases/download/1.1/deeplabv3_mobilenetv2_tf_dim_ordering_tf_kernels.h5"
WEIGHTS_PATH_X_CS = "https://github.com/bonlime/keras-deeplab-v3-plus/releases/download/1.2/deeplabv3_xception_tf_dim_ordering_tf_kernels_cityscapes.h5"
WEIGHTS_PATH_MOBILE_CS = "https://github.com/bonlime/keras-deeplab-v3-plus/releases/download/1.2/deeplabv3_mobilenetv2_tf_dim_ordering_tf_kernels_cityscapes.h5"


def Deeplabv3(weights_info={'weights': 'pascal_voc'}, input_tensor=None, input_shape=(512, 512, 3), classes=21, backbone='mobilenetv2',
              OS=16, alpha=1., activation='softmax', mask_dice_head=False):
    """ Instantiates the Deeplabv3+ architecture

    Optionally loads weights pre-trained
    on PASCAL VOC or Cityscapes. This model is available for TensorFlow only.
    # Arguments
        weights_info: this dict is consisted of `classes` and `weights`.
            `classes` is number of `weights` output units.
            `weights` is one of 'imagenet' (pre-training on ImageNet), 'pascal_voc', 'cityscapes',
            original weights path (pre-training on original data) or None (random initialization)
            `task` is one of 'classification' or 'segmentation'.
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: shape of input image. format HxWxC
            PASCAL VOC model was trained on (512,512,3) images. None is allowed as shape/width
        classes: number of desired classes. PASCAL VOC has 21 classes, Cityscapes has 19 classes.
            If number of classes not aligned with the weights used, last layer is initialized randomly
        backbone: backbone to use. one of {'xception','mobilenetv2'}
        activation: optional activation to add to the top of the network.
            One of 'softmax', 'sigmoid' or None
        OS: determines input_shape/feature_extractor_output ratio. One of {8,16}.
            Used only for xception backbone.
        alpha: controls the width of the MobileNetV2 network. This is known as the
            width multiplier in the MobileNetV2 paper.
                - If `alpha` < 1.0, proportionally decreases the number
                    of filters in each layer.
                - If `alpha` > 1.0, proportionally increases the number
                    of filters in each layer.
                - If `alpha` = 1, default number of filters from the paper
                    are used at each layer.
            Used only for mobilenetv2 backbone. Pretrained is only available for alpha=1.

    # Returns
        A Keras model instance.

    # Raises
        RuntimeError: If attempting to run this model with a
            backend that does not support separable convolutions.
        ValueError: in case of invalid argument for `weights` or `backbone`

    """

    if not (backbone in {'xception', 'mobilenetv2',
                         'efficientnetb0', 'efficientnetb1', 'efficientnetb2', 'efficientnetb3',
                         'efficientnetb4', 'efficientnetb5', 'efficientnetb6', 'efficientnetb7'}):
        raise ValueError('The `backbone` argument should be either '
                         '`xception`  or `mobilenetv2` or `efficientnetb#`')

    if weights_info is None:
        weights = None
    else:
        weights = weights_info.get("weights")
        if weights_info.get("task") == 'segmentation':
            output_classes = classes  # save variable `classes` in another variable
            # overwrite classes to load pretrained segmentation model
            classes = int(weights_info['classes'])

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        img_input = input_tensor

    if OS == 8:
        atrous_rates = (12, 24, 36)
    else:
        atrous_rates = (6, 12, 18)

    if backbone == 'xception':
        base_model, skip1 = DilatedXception(
            input_tensor=img_input,
            input_shape=input_shape,
            weights_info=weights_info,
            OS=OS,
            return_skip=True,
            include_top=False,
        )

    elif backbone.startswith('efficientnet'):
        model_name = backbone.replace('efficientnetb', 'EfficientNetB')
        base_model = getattr(efficientnet, model_name)(
            input_tensor=img_input,
            input_shape=input_shape,
            weights='imagenet',
            include_top=False,
        )

    else:
        base_model = MobileNetV2(
            input_tensor=img_input,
            input_shape=input_shape,
            weights_info=weights_info,
            OS=OS,
            include_top=False,
        )
    x = base_model.output
    # end of feature extractor

    # branching for Atrous Spatial Pyramid Pooling

    # Image Feature branch
    shape_before = tensorflow.shape(x)
    b4 = GlobalAveragePooling2D()(x)
    # from (b_size, channels)->(b_size, 1, 1, channels)
    b4 = Lambda(lambda x: tensorflow.keras.backend.expand_dims(x, 1))(b4)
    b4 = Lambda(lambda x: tensorflow.keras.backend.expand_dims(x, 1))(b4)
    b4 = Conv2D(256, (1, 1), padding='same',
                use_bias=False, name='image_pooling')(b4)
    b4 = BatchNormalization(name='image_pooling_BN', epsilon=1e-5)(b4)
    b4 = Activation('relu')(b4)
    # upsample. have to use compat because of the option align_corners
    size_before = tensorflow.keras.backend.int_shape(x)
    b4 = Lambda(lambda x: tensorflow.compat.v1.image.resize(x, size_before[1:3],
                                                            method='bilinear', align_corners=True))(b4)
    # simple 1x1
    b0 = Conv2D(256, (1, 1), padding='same', use_bias=False, name='aspp0')(x)
    b0 = BatchNormalization(name='aspp0_BN', epsilon=1e-5)(b0)
    b0 = Activation('relu', name='aspp0_activation')(b0)

    # there are only 2 branches in mobilenetV2. not sure why
    if backbone == 'mobilenetv2':
        x = Concatenate()([b4, b0])
    else:
        # rate = 6 (12)
        b1 = SepConv_BN(x, 256, 'aspp1',
                        rate=atrous_rates[0], depth_activation=True, epsilon=1e-5)
        # rate = 12 (24)
        b2 = SepConv_BN(x, 256, 'aspp2',
                        rate=atrous_rates[1], depth_activation=True, epsilon=1e-5)
        # rate = 18 (36)
        b3 = SepConv_BN(x, 256, 'aspp3',
                        rate=atrous_rates[2], depth_activation=True, epsilon=1e-5)

        # concatenate ASPP branches & project
        x = Concatenate()([b4, b0, b1, b2, b3])

    x = Conv2D(256, (1, 1), padding='same',
               use_bias=False, name='concat_projection')(x)
    x = BatchNormalization(name='concat_projection_BN', epsilon=1e-5)(x)
    x = Activation('relu')(x)
    x = Dropout(0.1, name='dropout_encoder_last')(x)

    # DeepLab v.3+ decoder
    if backbone == 'xception':
        # Feature projection
        # x4 (x2) block
        size_before2 = tensorflow.keras.backend.int_shape(x)
        x = Lambda(lambda xx: tensorflow.compat.v1.image.resize(xx,
                                                                size_before2[1:3] *
                                                                tensorflow.constant(OS // 4),
                                                                method='bilinear', align_corners=True))(x)

        dec_skip1 = Conv2D(48, (1, 1), padding='same',
                           use_bias=False, name='feature_projection0')(skip1)
        dec_skip1 = BatchNormalization(
            name='feature_projection0_BN', epsilon=1e-5)(dec_skip1)
        dec_skip1 = Activation('relu')(dec_skip1)
        x = Concatenate()([x, dec_skip1])
        x = SepConv_BN(x, 256, 'decoder_conv0',
                       depth_activation=True, epsilon=1e-5)
        x = SepConv_BN(x, 256, 'decoder_conv1',
                       depth_activation=True, epsilon=1e-5)

    # you can use it with arbitary number of classes
    if (weights == 'pascal_voc' and classes == 21) or (weights == 'cityscapes' and classes == 19):
        last_layer_name = 'logits_semantic'
    else:
        last_layer_name = 'custom_logits_semantic'

    x = Conv2D(classes, (1, 1), padding='same', name=last_layer_name)(x)
    size_before3 = tensorflow.keras.backend.int_shape(img_input)
    x = Lambda(lambda xx: tensorflow.compat.v1.image.resize(xx,
                                                            size_before3[1:3],
                                                            method='bilinear', align_corners=True))(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input

    if activation in {'softmax', 'sigmoid', 'linear'}:
        x = Activation(activation, name='last_activation')(x)

    model = Model(inputs, x, name='deeplabv3plus')

    if weights is None:
        return model

    # load weights
    if weights == 'pascal_voc':
        if backbone == 'xception':
            weights_path = get_file('deeplabv3_xception_tf_dim_ordering_tf_kernels.h5',
                                    WEIGHTS_PATH_X,
                                    cache_subdir='models')
        elif backbone == 'mobilenetv2':
            weights_path = get_file('deeplabv3_mobilenetv2_tf_dim_ordering_tf_kernels.h5',
                                    WEIGHTS_PATH_MOBILE,
                                    cache_subdir='models')
        else:
            return model
        model.load_weights(weights_path, by_name=True)
        print("loaded weights of pascal voc")

    elif weights == 'cityscapes':
        if backbone == 'xception':
            weights_path = get_file('deeplabv3_xception_tf_dim_ordering_tf_kernels_cityscapes.h5',
                                    WEIGHTS_PATH_X_CS,
                                    cache_subdir='models')
        elif backbone == 'mobilenetv2':
            weights_path = get_file('deeplabv3_mobilenetv2_tf_dim_ordering_tf_kernels_cityscapes.h5',
                                    WEIGHTS_PATH_MOBILE_CS,
                                    cache_subdir='models')
        else:
            return model
        model.load_weights(weights_path, by_name=True)
        print("loaded weights of cityscapes")

    elif os.path.exists(weights):
        if weights_info.get("task") == 'segmentation':
            model.load_weights(weights)
            # re-build to change output number of channels
            x = model.get_layer(index=-4).output  # before `custom_logits_semantic` layer
            x = Conv2D(output_classes, (1, 1), padding='same', name=last_layer_name)(x)
            x = Lambda(lambda xx: tensorflow.compat.v1.image.resize(xx,
                                                                    size_before3[1:3],
                                                                    method='bilinear', align_corners=True))(x)
            if activation in {'softmax', 'sigmoid', 'linear'}:
                x = Activation(activation)(x)

            model = Model(inputs=model.input, outputs=x, name='deeplabv3plus')
            print(f'loaded weights of {weights} and changed output number of classes from {classes} to {output_classes}')

    if mask_dice_head:
        # feature extractor
        encoder_last = model.get_layer(name='dropout_encoder_last').output  # (256, 512, 3) -> (16, 32, 256)
        # mask predictor
        segmentation_output = model.get_layer(name=last_layer_name).output  # (256, 512, 3) -> (64, 128, 256)
        segmentation_output = Activation(activation)(segmentation_output)
        x = MaxPool2D((4, 4), 4)(segmentation_output)  # (64, 128, 256) -> (16, 32, 256)
        # Dice regression head
        x = Concatenate()([x, encoder_last])
        for i in range(3):
            x = Conv2D(256, (3, 3), padding='same', use_bias=False)(x)
            x = BatchNormalization(epsilon=1e-5)(x)
            x = Activation('relu')(x)
        x = Conv2D(256, (3, 3), 2, padding='same', use_bias=False)(x)
        x = BatchNormalization(epsilon=1e-5)(x)
        x = Activation('relu')(x)
        x = Flatten()(x)
        x = Dense(1024, activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(1024, activation='relu')(x)
        x = Dropout(0.5)(x)
        # mask dice predictor
        x = Dense(classes)(x)
        x = ReLU(max_value=1, name='mask_dice_head')(x)

        model = Model(inputs=model.input, outputs=[model.output, x], name='deeplabv3plusWithDiceHead')
        print('build DeeplabV3+ with MaskDiceHead for regression dice')

    return model


def preprocess_input(x):
    """Preprocesses a numpy array encoding a batch of images.
    # Arguments
        x: a 4D numpy array consists of RGB values within [0, 255].
    # Returns
        Input array scaled to [-1.,1.]
    """
    return preprocess_input(x, mode='tf')
