import os
import pickle as pkl
from functools import reduce
import pprofile, io, contextlib
import shutil

import matplotlib.pyplot as plt
# import random
import numpy as np
import cv2
import matplotlib.patches as patches
import json

from operator_py.cython.bbox import bbox_overlaps_cython
from operator_py.bbox_transform import nonlinear_transform as bbox_transform


np.random.seed(5)


class NormParam:
    mean = tuple(i * 255 for i in (0.485, 0.456, 0.406)) # RGB order
    std = tuple(i * 255 for i in (0.229, 0.224, 0.225))

class ResizeParam:
    short = 800
    long = 1200

class PadParam:
    short = 800
    long = 1200
    max_num_gt = 100

class AnchorTarget2DParam:
    class generate:
        short = ResizeParam.short // 16
        long = ResizeParam.long // 16
        stride = 16
        scales = (2, 4, 8, 16, 32)
        aspects = (0.5, 1.0, 2.0)

    class assign:
        allowed_border = 0
        pos_thr = 0.7
        neg_thr = 0.3
        min_pos_thr = 0.0

    class sample:
        image_anchor = 256
        pos_fraction = 0.5

class DetectionAugmentation(object):
    def __init__(self):
        pass

    def apply(self, input_record):
        pass


class ReadRoiRecord(DetectionAugmentation):
    """
    input: image_url, str
           gt_url, str
    output: image, ndarray(h, w, rgb)
            image_raw_meta, tuple(h, w)
            gt, any
    """

    def __init__(self, gt_select):
        super().__init__()
        self.gt_select = gt_select

    def apply(self, input_record):
        image = cv2.imread(input_record["image_url"], cv2.IMREAD_COLOR)
        input_record["image"] = image[:, :, ::-1].astype("float32")
        # TODO: remove this compatibility method
        input_record["gt_bbox"] = np.concatenate([input_record["gt_bbox"],
                                                  input_record["gt_class"].reshape(-1, 1)],
                                                 axis=1)

        # gt_dict = pkl.load(input_record["gt_url"])
        # for s in self.gt_select:
        #     input_record[s] = gt_dict[s]


class Norm2DImage(DetectionAugmentation):
    """
    input: image, ndarray(h, w, rgb)
    output: image, ndarray(h, w, rgb)
    """

    def __init__(self, pNorm):
        super().__init__()
        self.p = pNorm  # type: NormParam

    def apply(self, input_record):
        p = self.p

        image = input_record["image"].astype(np.float32)

        image -= p.mean
        image /= p.std

        input_record["image"] = image


class Resize2DImageBbox(DetectionAugmentation):
    """
    input: image, ndarray(h, w, rgb)
           gt_bbox, ndarry(n, 5)
    output: image, ndarray(h', w', rgb)
            im_info, tuple(h', w', scale)
            gt_bbox, ndarray(n, 5)
    """

    def __init__(self, pResize):
        super().__init__()
        self.p = pResize  # type: ResizeParam

    def apply(self, input_record):
        p = self.p

        image = input_record["image"]
        gt_bbox = input_record["gt_bbox"].astype(np.float32)

        short = min(image.shape[:2])
        long = max(image.shape[:2])
        scale = min(p.short / short, p.long / long)

        input_record["image"] = cv2.resize(image, None, None, scale, scale,
                                           interpolation=cv2.INTER_LINEAR)
        # make sure gt boxes do not overflow
        gt_bbox[:, :4] = gt_bbox[:, :4] * scale
        if image.shape[0] < image.shape[1]:
            gt_bbox[:, [0, 2]] = np.clip(gt_bbox[:, [0, 2]], 0, p.long)
            gt_bbox[:, [1, 3]] = np.clip(gt_bbox[:, [1, 3]], 0, p.short)
        else:
            gt_bbox[:, [0, 2]] = np.clip(gt_bbox[:, [0, 2]], 0, p.short)
            gt_bbox[:, [1, 3]] = np.clip(gt_bbox[:, [1, 3]], 0, p.long)
        input_record["gt_bbox"] = gt_bbox

        # exactly as opencv
        h, w = image.shape[:2]
        input_record["im_info"] = np.array([round(h * scale), round(w * scale), scale], dtype=np.float32)


class Flip2DImageBbox(DetectionAugmentation):
    """
    input: image, ndarray(h, w, rgb)
           gt_bbox, ndarry(n, 4)
    output: image, ndarray(h, w, rgb)
            gt_bbox, ndarray(n, 4)
    """

    def __init__(self):
        super().__init__()

    def apply(self, input_record):
        if input_record["flipped"]:
            image = input_record["image"]
            gt_bbox = input_record["gt_bbox"]

            input_record["image"] = image[:, ::-1]
            flipped_bbox = gt_bbox.copy()
            h, w = image.shape[:2]
            flipped_bbox[:, 0] = (w - 1) - gt_bbox[:, 2]
            flipped_bbox[:, 2] = (w - 1) - gt_bbox[:, 0]
            input_record["gt_bbox"] = flipped_bbox


class Pad2DImageBbox(DetectionAugmentation):
    """
    input: image, ndarray(h, w, rgb)
           gt_bbox, ndarry(n, 5)
    output: image, ndarray(h, w, rgb)
            gt_bbox, ndarray(max_num_gt, 5)
    """

    def __init__(self, pPad):
        super().__init__()
        self.p = pPad  # type: PadParam

    def apply(self, input_record):
        p = self.p

        image = input_record["image"]
        gt_bbox = input_record["gt_bbox"]

        origin_h, origin_w = input_record["h"], input_record["w"]
        h, w = image.shape[:2]
        shape = (p.long, p.short, 3) if origin_h >= origin_w \
            else (p.short, p.long, 3)

        padded_image = np.zeros(shape, dtype=np.float32)

        padded_image[:h, :w] = image
        padded_gt_bbox = np.full(shape=(p.max_num_gt, 5), fill_value=-1, dtype=np.float32)
        padded_gt_bbox[:len(gt_bbox)] = gt_bbox

        input_record["image"] = padded_image
        input_record["gt_bbox"] = padded_gt_bbox


class ConvertImageFromHwcToChw(DetectionAugmentation):
    def __init__(self):
        super().__init__()

    def apply(self, input_record):
        input_record["image"] = input_record["image"].transpose((2, 0, 1))


class AnchorTarget2D(DetectionAugmentation):
    """
    input: image_meta: tuple(h, w, scale)
           gt_bbox, ndarry(max_num_gt, 5)
    output: anchor_label, ndarray(num_anchor * 2, h, w)
            anchor_bbox_target, ndarray(num_anchor * 4, h, w)
            anchor_bbox_weight, ndarray(num_anchor * 4, h, w)
    """

    def __init__(self, pAnchor):
        super().__init__()
        self.p = pAnchor  # type: AnchorTarget2DParam

        self.__base_anchor = None
        self.__v_all_anchor = None
        self.__h_all_anchor = None
        self.__num_anchor = None

        self.DEBUG = False

    @property
    def base_anchor(self):
        if self.__base_anchor is not None:
            return self.__base_anchor

        p = self.p

        base_anchor = np.array([0, 0, p.generate.stride - 1, self.p.generate.stride - 1])

        w = base_anchor[2] - base_anchor[0] + 1
        h = base_anchor[3] - base_anchor[1] + 1
        x_ctr = base_anchor[0] + 0.5 * (w - 1)
        y_ctr = base_anchor[1] + 0.5 * (h - 1)

        w_ratios = np.round(np.sqrt(w * h / p.generate.aspects))
        h_ratios = np.round(w_ratios * p.generate.aspects)
        ws = (np.outer(w_ratios, p.generate.scales)).reshape(-1)
        hs = (np.outer(h_ratios, p.generate.scales)).reshape(-1)

        base_anchor = np.stack(
            [x_ctr - 0.5 * (ws - 1),
             y_ctr - 0.5 * (hs - 1),
             x_ctr + 0.5 * (ws - 1),
             y_ctr + 0.5 * (hs - 1)],
            axis=1)

        self.__base_anchor = base_anchor
        return self.__base_anchor

    @property
    def v_all_anchor(self):
        if self.__v_all_anchor is not None:
            return self.__v_all_anchor

        p = self.p

        shift_x = np.arange(0, p.generate.short, dtype=np.float32) * p.generate.stride
        shift_y = np.arange(0, p.generate.long, dtype=np.float32) * p.generate.stride
        grid_x, grid_y = np.meshgrid(shift_x, shift_y)
        grid_x, grid_y = grid_x.reshape(-1), grid_y.reshape(-1)
        grid = np.stack([grid_x, grid_y, grid_x, grid_y], axis=1)
        all_anchor = grid[:, None, :] + self.base_anchor[None, :, :]
        all_anchor = all_anchor.reshape(-1, 4)

        self.__v_all_anchor = all_anchor
        self.__num_anchor = all_anchor.shape[0]
        return self.__v_all_anchor

    @property
    def h_all_anchor(self):
        if self.__h_all_anchor is not None:
            return self.__h_all_anchor

        p = self.p

        shift_x = np.arange(0, p.generate.long, dtype=np.float32) * p.generate.stride
        shift_y = np.arange(0, p.generate.short, dtype=np.float32) * p.generate.stride
        grid_x, grid_y = np.meshgrid(shift_x, shift_y)
        grid_x, grid_y = grid_x.reshape(-1), grid_y.reshape(-1)
        grid = np.stack([grid_x, grid_y, grid_x, grid_y], axis=1)
        all_anchor = grid[:, None, :] + self.base_anchor[None, :, :]
        all_anchor = all_anchor.reshape(-1, 4)

        self.__h_all_anchor = all_anchor
        self.__num_anchor = all_anchor.shape[0]
        return self.__h_all_anchor

    @v_all_anchor.setter
    def v_all_anchor(self, value):
        self.__v_all_anchor = value
        self.__num_anchor = value.shape[0]

    @h_all_anchor.setter
    def h_all_anchor(self, value):
        self.__h_all_anchor = value
        self.__num_anchor = value.shape[0]

    def _assign_label_to_anchor(self, valid_anchor, gt_bbox, neg_thr, pos_thr, min_pos_thr):
        num_anchor = valid_anchor.shape[0]
        cls_label = np.full(shape=(num_anchor,), fill_value=-1, dtype=np.float32)

        if len(gt_bbox) > 0:
            # num_anchor x num_gt
            overlaps = bbox_overlaps_cython(valid_anchor.astype(np.float32, copy=False), gt_bbox.astype(np.float32, copy=False))
            max_overlaps = overlaps.max(axis=1)
            argmax_overlaps = overlaps.argmax(axis=1)
            gt_max_overlaps = overlaps.max(axis=0)

            # TODO: speed up this
            # TODO: fix potentially assigning wrong anchors as positive
            # A correct implementation is given as
            # gt_argmax_overlaps = np.where((overlaps.transpose() == gt_max_overlaps[:, None]) &
            #                               (overlaps.transpose() >= min_pos_thr))[1]
            gt_argmax_overlaps = np.where((overlaps == gt_max_overlaps) &
                                          (overlaps >= min_pos_thr))[0]
            # anchor class
            cls_label[max_overlaps < neg_thr] = 0
            # fg label: for each gt, anchor with highest overlap
            cls_label[gt_argmax_overlaps] = 1
            # fg label: above threshold IoU
            cls_label[max_overlaps >= pos_thr] = 1
        else:
            cls_label[:] = 0
            argmax_overlaps = np.zeros(shape=(num_anchor, ))

        return cls_label, argmax_overlaps

    def _sample_anchor(self, label, num, fg_fraction):
        num_fg = int(fg_fraction * num)
        fg_inds = np.where(label == 1)[0]
        if len(fg_inds) > num_fg:
            disable_inds = np.random.choice(fg_inds, size=(len(fg_inds) - num_fg), replace=False)
            if self.DEBUG:
                disable_inds = fg_inds[:(len(fg_inds) - num_fg)]
            label[disable_inds] = -1

        num_bg = num - np.sum(label == 1)
        bg_inds = np.where(label == 0)[0]
        if len(bg_inds) > num_bg:
            disable_inds = np.random.choice(bg_inds, size=(len(bg_inds) - num_bg), replace=False)
            if self.DEBUG:
                disable_inds = bg_inds[:(len(bg_inds) - num_bg)]
            label[disable_inds] = -1

    def _cal_anchor_target(self, label, valid_anchor, gt_bbox, anchor_label):
        num_anchor = valid_anchor.shape[0]
        reg_target = np.zeros(shape=(num_anchor, 4), dtype=np.float32)
        reg_weight = np.zeros(shape=(num_anchor, 4), dtype=np.float32)
        fg_index = np.where(label == 1)[0]
        if len(fg_index) > 0:
            reg_target[fg_index] = bbox_transform(valid_anchor[fg_index], gt_bbox[anchor_label[fg_index], :4])
            reg_weight[fg_index, :] = 1.0

        return reg_target, reg_weight

    def _gather_valid_anchor(self, image_info):
        h, w = image_info[:2]
        all_anchor = self.v_all_anchor if h >= w else self.h_all_anchor
        allowed_border = self.p.assign.allowed_border
        valid_index = np.where((all_anchor[:, 0] >= -allowed_border) &
                               (all_anchor[:, 1] >= -allowed_border) &
                               (all_anchor[:, 2] < w + allowed_border) &
                               (all_anchor[:, 3] < h + allowed_border))[0]
        return valid_index, all_anchor[valid_index]

    def _scatter_valid_anchor(self, valid_index, cls_label, reg_target, reg_weight):
        num_anchor = self.__num_anchor

        all_cls_label = np.full(shape=(num_anchor,), fill_value=-1, dtype=np.float32)
        all_reg_target = np.zeros(shape=(num_anchor, 4), dtype=np.float32)
        all_reg_weight = np.zeros(shape=(num_anchor, 4), dtype=np.float32)

        all_cls_label[valid_index] = cls_label
        all_reg_target[valid_index] = reg_target
        all_reg_weight[valid_index] = reg_weight

        return all_cls_label, all_reg_target, all_reg_weight

    def apply(self, input_record):
        p = self.p

        im_info = input_record["im_info"]
        gt_bbox = input_record["gt_bbox"]
        assert isinstance(gt_bbox, np.ndarray)
        assert gt_bbox.dtype == np.float32
        valid = np.where(gt_bbox[:, 0] != -1)[0]
        gt_bbox = gt_bbox[valid]

        if gt_bbox.shape[1] == 5:
            gt_bbox = gt_bbox[:, :4]

        valid_index, valid_anchor = self._gather_valid_anchor(im_info)
        cls_label, anchor_label = \
            self._assign_label_to_anchor(valid_anchor, gt_bbox,
                                         p.assign.neg_thr, p.assign.pos_thr, p.assign.min_pos_thr)
        self._sample_anchor(cls_label, p.sample.image_anchor, p.sample.pos_fraction)
        reg_target, reg_weight = self._cal_anchor_target(cls_label, valid_anchor, gt_bbox, anchor_label)
        cls_label, reg_target, reg_weight = \
            self._scatter_valid_anchor(valid_index, cls_label, reg_target, reg_weight)

        h, w = im_info[:2]
        if h >= w:
            fh, fw = p.generate.long, p.generate.short
        else:
            fh, fw = p.generate.short, p.generate.long

        input_record["rpn_cls_label"] = cls_label.reshape((fh, fw, -1)).transpose(2, 0, 1).reshape(-1)
        input_record["rpn_reg_target"] = reg_target.reshape((fh, fw, -1)).transpose(2, 0, 1)
        input_record["rpn_reg_weight"] = reg_weight.reshape((fh, fw, -1)).transpose(2, 0, 1)

        return input_record["rpn_cls_label"], \
               input_record["rpn_reg_target"], \
               input_record["rpn_reg_weight"]



def do_transforms(roidb, transform):
    records = []
    for index in range(len(roidb)):
        roi_record = roidb[index].copy()
        for trans in transform:
            trans.apply(roi_record)
        records.append(roi_record)
    return records


def test_loader_tranform():

    transform = [
            ReadRoiRecord(None),
            # Norm2DImage(NormParam),
            Resize2DImageBbox(ResizeParam),
            # Flip2DImageBbox(),
            Pad2DImageBbox(PadParam),
            ConvertImageFromHwcToChw(),
            AnchorTarget2D(AnchorTarget2DParam)
        ]


    roidb = pkl.load(open(os.path.join("data/cache/coco_valminusminival2014.roidb"), "rb"), encoding="latin1")
    print("length of source roidb:{}".format(len(roidb)))
    roidb = roidb[:4]
    # transform = transform[:1]
    print(transform)

    # # add flip roi record
    # flipped_roidb = []
    # for rec in roidb:
    #     new_rec = rec.copy()
    #     new_rec["flipped"] = True
    #     flipped_roidb.append(new_rec)
    # roidb = roidb + flipped_roidb

    records = do_transforms(roidb, transform)
    print("keys:{}".format(records[0].keys()))

    # profile_name = "transform_profile.txt"
    # prof = pprofile.Profile()
    # with prof():
    #     print("*********** to profile producer {} *************".format(profile_name))
    #     records = do_transforms(roidb, transform)
    #     print("keys:{}".format(records[0].keys()))
    #     print("************profiled producer *****************")
    # fstr = io.StringIO()
    # with contextlib.redirect_stdout(fstr):
    #     prof.print_stats()
    # string = fstr.getvalue()
    # with open(profile_name, "w") as f:
    #     f.write(string)


def plt_image():
    transform = [
            ReadRoiRecord(None),
            # Norm2DImage(NormParam),
            Resize2DImageBbox(ResizeParam),
            Norm2DImage(NormParam),
            Pad2DImageBbox(PadParam),
            ConvertImageFromHwcToChw(),
        ]


    roidb = pkl.load(open("data/img_4.pkl", 'rb'))
    roidb = roidb[2:3]

    records = do_transforms(roidb, transform)

    for idx, record in enumerate(records):
        img_url = record["image_url"]
        print(img_url)

        img = record["image"]
        bboxes = record["gt_bbox"]
        bboxes = bboxes[bboxes[:,4] !=-1]
        labels = record["gt_class"]


        print(img.shape)
        print("img [max, min, mean, var]: [{}, {}, {}, {}]".format(np.max(img), np.min(img), np.mean(img), np.var(img)))
        print("img.type:{}, img[400,400]: {}".format(img.dtype, img[:,400,400]))
        # fig,ax = plt.subplots(1)
        # ax.imshow(img)


        # categories_set = set()
        # for label in labels:
        #     categories_set.add(label)

        # category_id_to_color = dict(
        #     [(cat_id, [np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1)]) for cat_id in categories_set])

        # for bbox, label in zip(bboxes, labels):
        #     rect = patches.Rectangle(
        #         (bbox[0], bbox[1]), # Absolute corner coordinates
        #         (bbox[2] - bbox[0]),    # Absolute bounding box width
        #         (bbox[3] - bbox[1]),    # Absolute bounding box height
        #         linewidth=1,
        #         edgecolor=category_id_to_color[label],
        #         facecolor='none')
        #     ax.add_patch(rect)
        # # plt.show()
        # plt.savefig("py_process_{}.pdf".format(idx))
        # plt.close()

def generate_20():
    image_list = ['COCO_val2014_000000262148.jpg',
                  'COCO_val2014_000000185686.jpg',
                  'COCO_val2014_000000360073.jpg',
                  'COCO_val2014_000000393225.jpg']
    val_json = json.load(open("data/small_ann/new_20.json"))

    small_val = {}
    small_val["categories"] = val_json["categories"]
    small_val["licenses"] = val_json["licenses"]
    small_val["info"] = val_json["info"]
    small_val["images"] = []
    small_val["annotations"] = []
    tmp_img_ids = {}
    
    for img in val_json["images"]:
        if img["file_name"] in image_list:
            print("{}:{}".format(img["id"], img["file_name"]))
            tmp_img_ids[img["id"]] = img["file_name"]
            small_val["images"].append(img)

    for ann in val_json["annotations"]:
        if ann['image_id'] in tmp_img_ids.keys():
            print("ann {} find".format(ann['image_id']))
            small_val["annotations"].append(ann)

    print("img ids len:{}".format(len(tmp_img_ids)))
    with open("new_4.json", 'w') as f:
        json.dump(small_val, f)


if __name__ == "__main__":
    test_loader_tranform()
    # plt_image()
    # generate_20()