import SimpleITK as sitk
import numpy as np
import SimpleITK as sitk
import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion
import os
path="./down_source/"

for dir,_,files in sorted(os.walk(path)):
    for filename in files:
        if "IP.mhd" in filename:
            ori = sitk.ReadImage("./down_source/"+filename, sitk.sitkFloat32)
            ori_np = sitk.GetArrayFromImage(ori)
            mask_np = sitk.GetArrayFromImage(sitk.ReadImage("./down_source/"+filename.strip(".mhd")+"total_mr.nii.gz"))

            W_np = sitk.GetArrayFromImage(sitk.ReadImage("./all_N4_224_W and F3/"+filename.strip("IP.mhd")+"W_mask.nii.gz"))
            F_np = sitk.GetArrayFromImage(sitk.ReadImage("./all_N4_224_W and F3/"+filename.strip("IP.mhd")+"F_mask.nii.gz"))

            # 器官标签
            organ_order = [
                ('spleen', 1),
                ('liver', 5),
                ('pancreas', 7),
                ('colon', 13, 15),
                ('vertebrae',19),
                ("trachea",16),
                ('intervertebral_discs', 20),
                ('heart', 22),
                ('left_kidney', 3),
                ('right_kidney', 2),
                ('stomach', 6),
                ('brain',56),
                ('aorta', 23)
            ]

            # 初始化
            processed_mask = np.zeros_like(mask_np)
            organ_dict = {}

            # 分步处理器官掩膜
            for idx, organ in enumerate(organ_order):
                name, *labels = organ
                organ_mask = np.zeros_like(mask_np)
                for l in labels:
                    organ_mask = np.logical_or(organ_mask, mask_np == l)
                if name=='liver' or name=='heart':
                    print(name)
                    organ_mask = binary_dilation(organ_mask, structure=np.ones((3,3,3)), iterations=1)
                if name=='trachea':
                    organ_mask = binary_erosion(organ_mask, structure=np.ones((3,3,3)), iterations=1)

                clean_dilated = np.where(processed_mask, 0, organ_mask)
                processed_mask = np.where(clean_dilated, idx + 1, processed_mask)
                print(idx+1,name)
                organ_dict[name] = clean_dilated.astype(np.uint8)

            # 修正水和脂肪掩膜
            W_np = np.where((W_np == 1) & (ori_np > 20) & (processed_mask == 0), 1, 0)
            F_np = np.where((F_np == 1) & (ori_np > 20)& (processed_mask == 0), 1, 0)

            processed_mask=np.where(processed_mask==6,14,processed_mask)
            processed_mask=np.where(W_np==1,6,processed_mask)
            processed_mask=np.where(F_np==1,15,processed_mask)
            final_mask = sitk.GetImageFromArray(processed_mask.astype(np.uint16))
            final_mask.CopyInformation(ori)
            sitk.WriteImage(final_mask, "./segmap/"+filename.strip(".mhd")+"seg_map.nii.gz")