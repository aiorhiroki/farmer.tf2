import tensorflow as tf
import segmentation_models
from segmentation_models.base import Loss
from ..losses import functional as F
from ..losses.functional import flooding, log_cosh
from segmentation_models.losses import CategoricalCELoss

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


class CategoricalFocalLoss(Loss):
    def __init__(self, alpha=0.25, gamma=2., class_weights=None, flooding_level=0.):
        super().__init__(name='categorical_focal_loss')
        self.alpha = alpha
        self.gamma = gamma
        self.class_weights = class_weights if class_weights is not None else 1.
        self.flooding_level = flooding_level

    @flooding
    def __call__(self, gt, pr):
        return F.categorical_focal_loss(
            gt,
            pr,
            alpha=self.alpha,
            gamma=self.gamma,
            class_weights=self.class_weights)


class DiceLoss(Loss):
    def __init__(self, beta=1, class_weights=None, flooding_level=50.):
        super().__init__(name='dice_loss')
        self.beta = beta
        self.class_weights = class_weights if class_weights is not None else 1
        self.flooding_level = flooding_level

    @flooding
    def __call__(self, gt, pr):
        return F.dice_loss(
            gt=gt,
            pr=pr,
            beta=self.beta,
            class_weights=self.class_weights)


class LogCoshDiceLoss(DiceLoss):
    def __init__(self, beta=1, class_weights=None, flooding_level=0.):
        # super().__init__
        super(LogCoshDiceLoss, self).__init__(
            beta=beta, class_weights=class_weights, flooding_level=flooding_level)
        # super().super().__init__
        super(DiceLoss, self).__init__(name='log_cosh_dice_loss')

    @flooding
    @log_cosh
    def __call__(self, gt, pr):
        # undecorate
        return super().__call__.__wrapped__(self, gt, pr)


class JaccardLoss(Loss):
    def __init__(self, class_weights=None, flooding_level=0.):
        super().__init__(name='jaccard_loss')
        self.class_weights = class_weights if class_weights is not None else 1
        self.flooding_level = flooding_level

    @flooding
    def __call__(self, gt, pr):
        return F.jaccard_loss(
            gt=gt,
            pr=pr,
            class_weights=self.class_weights)


class TverskyLoss(Loss):
    def __init__(self, alpha=0.45, beta=0.55, class_weights=None, flooding_level=0.):
        super().__init__(name='tversky_loss')
        self.alpha = alpha
        self.beta = beta
        self.class_weights = class_weights if class_weights is not None else 1.
        self.flooding_level = flooding_level

    @flooding
    def __call__(self, gt, pr):
        return F.tversky_loss(
            gt=gt,
            pr=pr,
            alpha=self.alpha,
            beta=self.beta,
            class_weights=self.class_weights)


class FocalTverskyLoss(Loss):
    def __init__(self, alpha=0.45, beta=0.55, gamma=2.5, class_weights=None, flooding_level=0.):
        super().__init__(name='focal_tversky_loss')
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.class_weights = class_weights if class_weights is not None else 1.
        self.flooding_level = flooding_level

    @flooding
    def __call__(self, gt, pr):
        return F.focal_tversky_loss(
            gt=gt,
            pr=pr,
            alpha=self.alpha,
            beta=self.beta,
            gamma=self.gamma,
            class_weights=self.class_weights)


class LogCoshTverskyLoss(TverskyLoss):
    def __init__(self, alpha=0.3, beta=0.7, class_weights=None, flooding_level=0.):
        # super().__init__
        super(LogCoshTverskyLoss, self).__init__(
            alpha=alpha, beta=beta, class_weights=class_weights, flooding_level=flooding_level)
        # super().super().__init__
        super(TverskyLoss, self).__init__(name='log_cosh_tversky_loss')

    @flooding
    @log_cosh
    def __call__(self, gt, pr):
        # undecorate
        return super().__call__.__wrapped__(self, gt, pr)


class LogCoshFocalTverskyLoss(FocalTverskyLoss):
    def __init__(self, alpha=0.3, beta=0.7, gamma=1.3, class_weights=None, flooding_level=0.):
        # super().__init__
        super(LogCoshFocalTverskyLoss, self).__init__(
            alpha=alpha, beta=beta, gamma=gamma, class_weights=class_weights, flooding_level=flooding_level)
        # super().super().__init__
        super(FocalTverskyLoss, self).__init__(name='log_cosh_focal_tversky_loss')

    @flooding
    @log_cosh
    def __call__(self, gt, pr):
        # undecorate
        return super().__call__.__wrapped__(self, gt, pr)


class LogCoshLoss(Loss):
    def __init__(self, base_loss, flooding_level=0., **kwargs):
        super().__init__(name=f'log_cosh_{base_loss}')
        self.loss = getattr(F, base_loss)
        self.flooding_level = flooding_level
        self.kwargs = kwargs

    @flooding
    def __call__(self, gt, pr):
        x = self.loss(gt, pr, **self.kwargs)
        return tf.math.log((tf.exp(x) + tf.exp(-x)) / 2.0)


class BoundaryLoss(Loss):
    def __init__(self, flooding_level=0., **kwargs):
        super().__init__(name='boundary_loss')
        self.flooding_level = flooding_level

    @flooding
    def __call__(self, gt, pr):
        return F.surface_loss(
            gt=gt,
            pr=pr)


class UnifiedFocalLoss(Loss):
    def __init__(self, weight=0.5, delta=0.6, gamma=0.2, flooding_level=0.):
        super().__init__(name='unified_focal_loss')
        self.weight = weight
        self.delta = delta
        self.gamma = gamma
        self.flooding_level = flooding_level

    @flooding
    def __call__(self, gt, pr):
        return F.unified_focal_loss(
            gt=gt,
            pr=pr,
            weight=self.weight,
            delta=self.delta,
            gamma=self.gamma)


class MccLoss(Loss):
    def __init__(self, class_weights=None, flooding_level=0.):
        super().__init__(name='mcc_loss')
        self.class_weights = class_weights if class_weights is not None else 1
        self.flooding_level = flooding_level

    @flooding
    def __call__(self, gt, pr):
        return F.mcc_loss(
            gt=gt,
            pr=pr,
            class_weights=self.class_weights)


class FocalPhiLoss(Loss):
    def __init__(self, gamma=1.5, class_weights=None, flooding_level=0.):
        super().__init__(name='focal_phi_loss')
        self.gamma = gamma
        self.class_weights = class_weights if class_weights is not None else 1
        self.flooding_level = flooding_level

    @flooding
    def __call__(self, gt, pr):
        return F.focal_phi_loss(
            gt=gt,
            pr=pr,
            gamma=self.gamma,
            class_weights=self.class_weights)



class ActiveContourLoss(Loss):
    def __init__(self, w_region=1.0, w_region_in=1.0, w_region_out=1.0, flooding_level=0.):
        super().__init__(name='active_contour_loss')
        self.w_region = tf.Variable(w_region, dtype=tf.float32) # lambda in the paper
        self.w_region_in = tf.Variable(w_region_in, dtype=tf.float32)
        self.w_region_out = tf.Variable(w_region_out, dtype=tf.float32)
        self.flooding_level = tf.Variable(flooding_level, dtype=tf.float32)
    
    @flooding
    def __call__(self, gt, pr):
        return F.active_contour_loss(
            gt=gt,
            pr=pr,
            w_region=self.w_region,
            w_region_in=self.w_region_in,
            w_region_out=self.w_region_out)
