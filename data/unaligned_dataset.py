import os
import random
import numpy as np
import torch
import torch.multiprocessing as mp

from data.base_dataset import BaseDataset
from data.image_folder import make_dataset


class UnalignedDataset(BaseDataset):
    def __init__(self, opt):
        self.opt = opt
        self.target_size = tuple(opt.imgSize)   # (D, H, W)
        self.semi_supervised = getattr(opt, "semi_supervised", False)
        self.semi_warmup_iters = getattr(opt, "semi_warmup_iters", 0)
        self.num_labeled = getattr(opt, "num_labeled", 0)
        self.unlabeled_ratio = getattr(opt, "unlabeled_ratio", 0.5)
        print("unlabeled_ratio",self.unlabeled_ratio)
        self.dir_img = os.path.join(opt.dataroot, opt.phase, "img")
        self.dir_bf = os.path.join(opt.dataroot, opt.phase, "bf_w_f_200")
        self.dir_N4WI = os.path.join(opt.dataroot, opt.phase, "N4WI_w_f_200")
        self.dir_fm = os.path.join(opt.dataroot, opt.phase, "F_mask")
        self.dir_wm = os.path.join(opt.dataroot, opt.phase, "W_mask")
        self.dir_m = os.path.join(opt.dataroot, opt.phase, "mask")
        self.dir_tissue_norm_mask = os.path.join(opt.dataroot, opt.phase, f"seg_map_{opt.phase}")

        self.img_paths = sorted(make_dataset(self.dir_img, opt.max_dataset_size))
        self.bf_paths = sorted(make_dataset(self.dir_bf, opt.max_dataset_size))
        self.N4WI_paths = sorted(make_dataset(self.dir_N4WI, opt.max_dataset_size))
        self.fm_paths = sorted(make_dataset(self.dir_fm, opt.max_dataset_size))
        self.wm_paths = sorted(make_dataset(self.dir_wm, opt.max_dataset_size))
        self.m_paths = sorted(make_dataset(self.dir_m, opt.max_dataset_size))
        self.tissue_norm_mask_paths = sorted(make_dataset(self.dir_tissue_norm_mask, opt.max_dataset_size))

        self.dataset_size = len(self.img_paths)

        self.iter_counter = mp.Value('i', 0)
        self.lock = mp.Lock()

        # index split
        self.labeled_indices = list(range(min(self.num_labeled, self.dataset_size)))
        self.unlabeled_indices = list(range(min(self.num_labeled, self.dataset_size), self.dataset_size))
        print("indices",self.labeled_indices)
        print("unindices",self.unlabeled_indices)
        print("self.semi_supervised",self.semi_supervised)
        print("self.semi_warmup_iters",self.semi_warmup_iters)
    def __len__(self):
        return self.dataset_size

    def _sample_index(self, index):
        if self.opt.phase != "train" or not self.semi_supervised:
            return index % self.dataset_size, True

        with self.lock:
            current_iter = self.iter_counter.value

        # warmup stage: only labeled samples
        if current_iter < self.semi_warmup_iters:
            sampled_index = random.choice(self.labeled_indices)
            return sampled_index, True

        # semi-supervised stage
        use_unlabeled = (
            len(self.unlabeled_indices) > 0 and
            random.random() < self.unlabeled_ratio
        )

        if use_unlabeled:
            sampled_index = random.choice(self.unlabeled_indices)
            return sampled_index, False
        else:
            sampled_index = random.choice(self.labeled_indices)
            return sampled_index, True

    def _load_npy(self, path):
        return np.load(path)

    def _pad_or_crop(self, arr, target_shape, pad_value=0):
        d, h, w = arr.shape
        td, th, tw = target_shape

        arr = arr[:td, :th, :tw]

        pd = max(0, td - arr.shape[0])
        ph = max(0, th - arr.shape[1])
        pw = max(0, tw - arr.shape[2])

        arr = np.pad(
            arr,
            ((0, pd), (0, ph), (0, pw)),
            mode='constant',
            constant_values=pad_value
        )
        return arr

    def _compute_water_peak(self, img):
        nonzero = img[np.nonzero(img)].ravel()
        if len(nonzero) == 0:
            return 1.0

        hist, bins = np.histogram(nonzero, bins=50, density=True)
        hist_s = np.convolve(hist, np.ones(3) / 3, mode='same')
        water_peak = self.findWater(hist_s, bins)
        if water_peak is False or water_peak == 0:
            water_peak = 1.0
        return float(water_peak)

    def _one_hot_mask(self, mask, num_classes, target_shape):
        mask_map = (mask[..., None] == np.arange(num_classes)).astype(np.uint8)
        mask_map = np.moveaxis(mask_map, -1, 0)[1:]  # remove background

        padded = np.zeros((mask_map.shape[0], *target_shape), dtype=np.uint8)
        for c in range(mask_map.shape[0]):
            padded[c] = self._pad_or_crop(mask_map[c], target_shape, pad_value=0)
        return padded

    def findWater(self, hist, bin_edges):
        peak = 0
        descend = True
        class_number_pre = hist[0]
        i = 0
        for class_number in hist:
            if i < 4:
                i += 1
                continue
            if class_number_pre >= class_number:
                if descend is False:
                    peak += 1
                    if bin_edges[i] >= 50:
                        return (bin_edges[i] + bin_edges[i - 1]) / 2
                descend = True
            else:
                descend = False
            class_number_pre = class_number
            i += 1
            if i > 30:
                break
        return False

    def __getitem__(self, index):
        index, is_labeled = self._sample_index(index)

        img_path = self.img_paths[index]
        bf_path = self.bf_paths[index]
        N4WI_path = self.N4WI_paths[index]
        fm_path = self.fm_paths[index]
        wm_path = self.wm_paths[index]
        m_path = self.m_paths[index]
        tissue_norm_mask_path = self.tissue_norm_mask_paths[index]
        print(img_path)
        print(bf_path)
        print(N4WI_path)
        print(fm_path)
        print(wm_path)
        print(m_path)
        print(tissue_norm_mask_path)
        bf_img = self._load_npy(bf_path)
        N4WI_img = self._load_npy(N4WI_path)
        fm_img = self._load_npy(fm_path)
        wm_img = self._load_npy(wm_path)
        tissue_norm_mask = self._load_npy(tissue_norm_mask_path)

        ori_img = N4WI_img * bf_img
        m_img = np.where(ori_img > 25, 1, 0)
        wm_img=np.where(m_img==1,wm_img,0)
        fm_img=np.where(m_img==1,fm_img,0)
        img = np.where(m_img==1, ori_img, 0)
        tissue_norm_mask=np.where(m_img==1,tissue_norm_mask,0)
        water_peak = self._compute_water_peak(img)
        print(water_peak)
        ori_img = ori_img / water_peak
        N4WI_img = N4WI_img / water_peak

        original_shape = ori_img.shape

        ori_img = self._pad_or_crop(ori_img, self.target_size, pad_value=0)
        N4WI_img = self._pad_or_crop(N4WI_img, self.target_size, pad_value=0)
        bf_img = self._pad_or_crop(bf_img, self.target_size, pad_value=1)
        fm_img = self._pad_or_crop(fm_img, self.target_size, pad_value=0)
        wm_img = self._pad_or_crop(wm_img, self.target_size, pad_value=0)
        m_img = self._pad_or_crop(m_img, self.target_size, pad_value=0)

        if is_labeled:
            tissue_norm_mask_map = self._one_hot_mask(
                tissue_norm_mask,
                self.opt.num_classes,
                self.target_size
            )
        else:
            tissue_norm_mask_map = np.zeros(
                (self.opt.num_classes - 1, *self.target_size),
                dtype=np.uint8
            )

        sample = {
            "is_labeled": torch.tensor(is_labeled, dtype=torch.bool),
            "Ori": torch.from_numpy(ori_img).unsqueeze(0).float(),
            "BF": torch.from_numpy(bf_img).unsqueeze(0).float(),
            "N4WI": torch.from_numpy(N4WI_img).unsqueeze(0).float(),
            "FM": torch.from_numpy(fm_img).unsqueeze(0).float(),
            "WM": torch.from_numpy(wm_img).unsqueeze(0).float(),
            "M": torch.from_numpy(m_img).unsqueeze(0).float(),
            "TissueNormM": torch.from_numpy(tissue_norm_mask_map).float(),
            "W_peak": torch.tensor([water_peak], dtype=torch.float32),
            "Image_shape": torch.tensor(original_shape, dtype=torch.long),
            "img_paths": img_path,
            "bf_paths": bf_path,
            "N4WI_paths": N4WI_path,
            "fm_paths": fm_path,
            "wm_paths": wm_path,
            "m_paths": m_path,
            "tissue_norm_mask_paths": tissue_norm_mask_path,
        }

        with self.lock:
            self.iter_counter.value += 1

        return sample