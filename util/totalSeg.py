import SimpleITK
import nibabel as nib
from totalsegmentator.python_api import totalsegmentator
# from totalsegmentator.nnunet import
import os
import torch
if __name__ == '__main__':
    filePath="./source"
    for dir, _, files in sorted(os.walk(filePath)):
        for file in files:
            if "IP.mhd" in file:
                input_img = nib.load(dir+"/"+file)
                output_img_tissue = totalsegmentator(input_img,device="gpu",task="total_mr")
                # output_img = totalsegmentator(input_img, device="gpu", task="total_mr")
                nib.save(output_img_tissue,"./seg"+"/"+file.strip(".mhd")+"total_mr.nii.gz")
                torch.cuda.empty_cache()

