import cv2
import numpy as np


def create_isolated_mask(pr, gt, label_id, output='fp'):
    """
    Create a mask image by extracting only the isoated objects that are not overlaped by one of the masks.

    Args:
        pr ([type]): predicted mask
        gt ([type]): GT mask
        output (str, optional): Type of mask to be created. 'fp', 'fn' can be specified . Defaults to 'fp'.

    Returns:
        [np.array]: created mask.
                    If output is 'fp', create a mask consisting of FP objects that are not overlaped by GT objects.
                    If output is 'fn', create a mask consisting of FN objects that are not overlaped by predicted objects.
                    If output is not specified correctly, return None.
    """
    if output == 'fp':
        # If the output is fp, remove predicted objects that overlap with GT objects.
        mask_src = pr.copy()
        mask_tar = gt.copy()
    elif output == 'fn':
        # If the output is fn, remove GT　objects that overlap with predicted objects.
        mask_src = gt.copy()
        mask_tar = pr.copy()
    else:
        print('[ERR] unexpected arg output')
        return None

    mask_src[mask_src != label_id] = 0
    mask_tar[mask_tar != label_id] = 0

    mask_src_tmp = mask_src.copy()
    mask_tar_tmp = mask_tar.copy()

    n_label_src, label_im_src = cv2.connectedComponents(mask_src_tmp)
    n_label_tar, label_im_tar = cv2.connectedComponents(mask_tar_tmp)

    # Skipping label_id:0 and start the loop from label_id:1
    for label_id_src in range(1, n_label_src):
        src_obj = label_im_src == label_id_src

        for label_id_tar in range(1, n_label_tar):
            tar_obj = label_im_tar == label_id_tar
            duplicated_area = np.sum(src_obj * tar_obj)

            if duplicated_area > 0:
                mask_src[src_obj] = 0

    return mask_src


def calc_isolated_fp(pred_out, gt_label, nb_classes):
    pred_label = np.uint8(np.argmax(pred_out, axis=2))
    gt_label = np.uint8(np.argmax(gt_label, axis=2))
    isolated_fp = list()

    for class_id in range(nb_classes):
        gt_mask = np.asarray(gt_label == class_id, dtype=np.int8)
        pred_mask = np.asarray(pred_label == class_id, dtype=np.int8)
        
        isolated_fp_mask = create_isolated_mask(pred_mask, gt_mask, label_id=1, output='fp')
        isolated_fp.append(np.sum(isolated_fp_mask))
    
    return np.asarray(isolated_fp)
