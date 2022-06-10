import tensorflow as tf
import segmentation_models
from segmentation_models.base import Loss
from segmentation_models.losses import CategoricalCELoss
from ..losses import functional as F

segmentation_models.set_framework('tf.keras')

class CompoundLoss(Loss):
    def __init__(self, l1, l2, w_l1=1., w_l2=1.):
        name = '{}_plus_{}'.format(l1.__name__, l2.__name__)
        super().__init__(name=name)
        self.l1 = l1
        self.l2 = l2
        self.w_l1 = tf.Variable(w_l1, dtype=tf.float32)
        self.w_l2 = tf.Variable(w_l2, dtype=tf.float32)

    def __call__(self, gt, pr):
        return self.w_l1 * self.l1(gt, pr) + self.w_l2 * self.l2(gt, pr)

class DiceLoss(Loss):
    def __init__(self, beta=1, class_weights=None, flooding_level=0.):
        super().__init__(name='dice_loss')
        self.beta = tf.Variable(beta, dtype=tf.float32)
        if class_weights is not None:
            self.class_weights = tf.Variable(class_weights, dtype=tf.float32) 
        else:
            self.class_weights = tf.Variable(1, dtype=tf.float32)
        self.flooding_level = tf.Variable(flooding_level, dtype=tf.float32)

    def __call__(self, gt, pr):
        return F.flooding(F.dice_loss(
            gt=gt,
            pr=pr,
            beta=self.beta,
            class_weights=self.class_weights
        ), self.flooding_level)


class JaccardLoss(Loss):
    def __init__(self, class_weights=None, flooding_level=0.):
        super().__init__(name='jaccard_loss')
        if class_weights is not None:
            self.class_weights = tf.Variable(class_weights, dtype=tf.float32) 
        else:
            self.class_weights = tf.Variable(1, dtype=tf.float32)
        self.flooding_level = tf.Variable(flooding_level, dtype=tf.float32)

    def __call__(self, gt, pr):
        return F.flooding(F.jaccard_loss(
            gt=gt,
            pr=pr,
            class_weights=self.class_weights
        ), self.flooding_level)


class TverskyLoss(Loss):
    def __init__(self, alpha=0.45, beta=0.55, class_weights=None, flooding_level=0.):
        super().__init__(name='tversky_loss')
        self.alpha = tf.Variable(alpha, dtype=tf.float32)
        self.beta = tf.Variable(beta, dtype=tf.float32)
        if class_weights is not None:
            self.class_weights = tf.Variable(class_weights, dtype=tf.float32) 
        else:
            self.class_weights = tf.Variable(1, dtype=tf.float32)
        self.flooding_level = tf.Variable(flooding_level, dtype=tf.float32)

    def __call__(self, gt, pr):
        return F.flooding(F.tversky_loss(
            gt=gt,
            pr=pr,
            alpha=self.alpha,
            beta=self.beta,
            class_weights=self.class_weights
        ), self.flooding_level)


class FocalTverskyLoss(Loss):
    def __init__(self, alpha=0.45, beta=0.55, gamma=2.5, class_weights=None, flooding_level=0.):
        super().__init__(name='focal_tversky_loss')
        self.alpha = tf.Variable(alpha, dtype=tf.float32)
        self.beta = tf.Variable(beta, dtype=tf.float32)
        self.gamma = tf.Variable(gamma, dtype=tf.float32)
        if class_weights is not None:
            self.class_weights = tf.Variable(class_weights, dtype=tf.float32) 
        else:
            self.class_weights = tf.Variable(1, dtype=tf.float32)
        self.flooding_level = tf.Variable(flooding_level, dtype=tf.float32)

    def __call__(self, gt, pr):
        return F.flooding(F.focal_tversky_loss(
            gt=gt,
            pr=pr,
            alpha=self.alpha,
            beta=self.beta,
            gamma=self.gamma,
            class_weights=self.class_weights
        ), self.flooding_level)


class CategoricalFocalLoss(Loss):
    def __init__(self, alpha=0.25, gamma=2., class_weights=None, flooding_level=0.):
        super().__init__(name='categorical_focal_loss')
        self.alpha = tf.Variable(alpha, dtype=tf.float32)
        self.gamma = tf.Variable(gamma, dtype=tf.float32)
        if class_weights is not None:
            self.class_weights = tf.Variable(class_weights, dtype=tf.float32) 
        else:
            self.class_weights = tf.Variable(1, dtype=tf.float32)
        self.flooding_level = tf.Variable(flooding_level, dtype=tf.float32)

    def __call__(self, gt, pr):
        return F.flooding(F.categorical_focal_loss(
            gt,
            pr,
            alpha=self.alpha,
            gamma=self.gamma,
            class_weights=self.class_weights
        ), self.flooding_level)


class LogCoshDiceLoss(Loss):
    def __init__(self, beta=1, class_weights=None, flooding_level=0.):
        super().__init__(name='log_cosh_dice_loss')
        self.beta = tf.Variable(beta, dtype=tf.float32)
        if class_weights is not None:
            self.class_weights = tf.Variable(class_weights, dtype=tf.float32) 
        else:
            self.class_weights = tf.Variable(1, dtype=tf.float32)
        self.flooding_level = tf.Variable(flooding_level, dtype=tf.float32)

    def __call__(self, gt, pr):
        return F.flooding(F.log_cosh_dice_loss(
            gt=gt,
            pr=pr,
            beta=self.beta,
            class_weights=self.class_weights
        ), self.flooding_level)


class LogCoshTverskyLoss(Loss):
    def __init__(self, alpha=0.3, beta=0.7, class_weights=None, flooding_level=0.):
        super().__init__(name='log_cosh_tversky_loss')
        self.alpha = tf.Variable(alpha, dtype=tf.float32)
        self.beta = tf.Variable(beta, dtype=tf.float32)
        if class_weights is not None:
            self.class_weights = tf.Variable(class_weights, dtype=tf.float32) 
        else:
            self.class_weights = tf.Variable(1, dtype=tf.float32)
        self.flooding_level = tf.Variable(flooding_level, dtype=tf.float32)

    def __call__(self, gt, pr):
        return F.flooding(F.log_cosh_tversky_loss(
            gt=gt,
            pr=pr,
            alpha=self.alpha,
            beta=self.beta,
            class_weights=self.class_weights
        ), self.flooding_level)


class LogCoshFocalTverskyLoss(Loss):
    def __init__(self, alpha=0.3, beta=0.7, gamma=1.3, class_weights=None, flooding_level=0.):
        super().__init__(name='log_cosh_focal_tversky_loss')
        self.alpha = tf.Variable(alpha, dtype=tf.float32)
        self.beta = tf.Variable(beta, dtype=tf.float32)
        self.gamma = tf.Variable(gamma, dtype=tf.float32)
        if class_weights is not None:
            self.class_weights = tf.Variable(class_weights, dtype=tf.float32) 
        else:
            self.class_weights = tf.Variable(1, dtype=tf.float32)
        self.flooding_level = tf.Variable(flooding_level, dtype=tf.float32)

    def __call__(self, gt, pr):
        return F.flooding(F.log_cosh_focal_tversky_loss(
            gt=gt,
            pr=pr,
            alpha=self.alpha,
            beta=self.beta,
            gamma=self.gamma,
            class_weights=self.class_weights
        ), self.flooding_level)


class LogCoshLoss(Loss):
    def __init__(self, base_loss, flooding_level=0., **kwargs):
        super().__init__(name=f'log_cosh_{base_loss}')
        self.loss = getattr(F, base_loss)
        self.flooding_level = tf.Variable(flooding_level, dtype=tf.float32)
        self.kwargs = kwargs

    def __call__(self, gt, pr):
        x = self.loss(gt, pr, **self.kwargs)
        return F.flooding(
            tf.math.log((tf.exp(x) + tf.exp(-x)) / 2.0),
            self.flooding_level)


class BoundaryLoss(Loss):
    def __init__(self, flooding_level=0., **kwargs):
        super().__init__(name='boundary_loss')
        self.flooding_level = tf.Variable(flooding_level, dtype=tf.float32)

    def __call__(self, gt, pr):
        return F.flooding(F.surface_loss(
            gt=gt,
            pr=pr,
        ), self.flooding_level)

class UnifiedFocalLoss(Loss):
    def __init__(self, weight=0.5, delta=0.6, gamma=0.2, flooding_level=0.):
        super().__init__(name='unified_focal_loss')
        self.weight = tf.Variable(weight, dtype=tf.float32)
        self.delta = tf.Variable(delta, dtype=tf.float32)
        self.gamma = tf.Variable(gamma, dtype=tf.float32)
        self.flooding_level = tf.Variable(flooding_level, dtype=tf.float32)

    def __call__(self, gt, pr):
        return F.flooding(F.unified_focal_loss(
            gt=gt,
            pr=pr,
            weight=self.weight,
            delta=self.delta,
            gamma=self.gamma
            ), self.flooding_level)

class MccLoss(Loss):
    def __init__(self, class_weights=None, flooding_level=0.):
        super().__init__(name='mcc_loss')
        self.class_weights = class_weights if class_weights is not None else 1
        self.flooding_level = flooding_level

    def __call__(self, gt, pr):
        return F.flooding(
            F.mcc_loss(
            gt=gt,
            pr=pr,
            class_weights=self.class_weights
        ), self.flooding_level)

class FocalPhiLoss(Loss):
    def __init__(self, gamma=1.5, class_weights=None, flooding_level=0.):
        super().__init__(name='focal_phi_loss')
        self.gamma = gamma
        self.class_weights = class_weights if class_weights is not None else 1
        self.flooding_level = flooding_level

    def __call__(self, gt, pr):
        return F.flooding(
            F.focal_phi_loss(
            gt=gt,
            pr=pr,
            gamma=self.gamma,
            class_weights=self.class_weights
        ), self.flooding_level)

class ActiveContourLoss(Loss):
    def __init__(self, w_region=1.0, w_region_in=1.0, w_region_out=1.0, flooding_level=0.):
        super().__init__(name='active_contour_loss')
        self.w_region = tf.Variable(w_region, dtype=tf.float32) # lambda in the paper
        self.w_region_in = tf.Variable(w_region_in, dtype=tf.float32)
        self.w_region_out = tf.Variable(w_region_out, dtype=tf.float32)
        self.flooding_level = tf.Variable(flooding_level, dtype=tf.float32)
    
    def __call__(self, gt, pr):
        return F.flooding(F.active_contour_loss(
            gt=gt,
            pr=pr,
            w_region=self.w_region,
            w_region_in=self.w_region_in,
            w_region_out=self.w_region_out,
        ), self.flooding_level)

class RecallLoss(Loss):
    def __init__(self, class_weights=None, flooding_level=0.):
        super().__init__(name='recall_loss')
        self.class_weights = class_weights if class_weights is not None else 1
        self.flooding_level = tf.Variable(flooding_level, dtype=tf.float32)
    
    def __call__(self, gt, pr):
        return F.flooding(F.recall_loss(
            gt=gt,
            pr=pr,
            class_weights=self.class_weights
        ), self.flooding_level)

class FocalRecallLoss(Loss):
    def __init__(self, gamma=2.0, alpha=0.25, class_weights=None, flooding_level=0.):
        super().__init__(name='focal_recall_loss')
        self.gamma = gamma
        self.alpha = alpha
        self.class_weights = class_weights if class_weights is not None else 1
        self.flooding_level = tf.Variable(flooding_level, dtype=tf.float32)
    
    def __call__(self, gt, pr):
        return F.flooding(F.focal_recall_loss(
            gt,
            pr,
            gamma=self.gamma,
            alpha=self.alpha,
            class_weights=self.class_weights
        ), self.flooding_level)

class Poly1Loss(Loss):
    def __init__(self, epsilon=-1, class_weights=None, flooding_level=0.):
        super().__init__(name='poly1_loss')
        self.epsilon = epsilon
        self.class_weights = class_weights if class_weights is not None else 1
        self.flooding_level = tf.Variable(flooding_level, dtype=tf.float32)
    
    def __call__(self, gt, pr):
        return F.flooding(F.poly1_loss(
            gt=gt,
            pr=pr,
            epsilon=self.epsilon,
            class_weights=self.class_weights
        ), self.flooding_level)

class Poly1FocalLoss(Loss):
    def __init__(self, epsilon=-1, gamma=2.0, alpha=0.25, class_weights=None, flooding_level=0.):
        super().__init__(name='poly1_focal_loss')
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha
        self.class_weights = class_weights if class_weights is not None else 1
        self.flooding_level = tf.Variable(flooding_level, dtype=tf.float32)
    
    def __call__(self, gt, pr):
        return F.flooding(F.poly1_focal_loss(
            gt=gt,
            pr=pr,
            epsilon=self.epsilon,
            gamma=self.gamma,
            alpha=self.alpha,
            class_weights=self.class_weights
        ), self.flooding_level)
