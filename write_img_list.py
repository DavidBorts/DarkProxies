import sys
import os

import Constants as c
from misc import write_img_list

name = sys.argv[1]

output_path = os.path.join(c.IMAGE_ROOT_DIR, c.STAGE_1_PATH, name + '_' + c.OUTPUT_DIR)
gt_imgs = os.listdir(output_path)
write_img_list(name, 1, gt_imgs)
