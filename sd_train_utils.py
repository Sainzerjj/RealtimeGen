import torch
from PIL import Image
import PIL.Image
import numpy as np
import json
import os
from datetime import datetime
def save_file(denoised_images: list[PIL.Image.Image], x_t: PIL.Image.Image, type: str, settings: dict):
    # image_files = [f for f in os.listdir(output) 
    #                 if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]
    # 遍历对应文件夹下所有图片转存到另一个位置, 并删除原文件夹
    image_ls = ["denosed_prediction", "x_t"]
    if type == "original":
        root_dir = os.path.join("/data/AI_paintings/imgs/", "ori_imgs")
    else:
        root_dir = os.path.join("/data/AI_paintings/imgs/", "edit_imgs")
    current_time = datetime.now()
    hashcode = current_time.strftime('%Y%m%d%H%M%S')
    for index in range(2):
        target_dir = os.path.join(root_dir, hashcode, image_ls[index])
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        if index == 0:
            for i, image in enumerate(denoised_images):
                image.save(os.path.join(target_dir, str(i) + ".png"))
        else:
            # 将JSON字符串写入文件
            json_path = os.path.join(target_dir, "data.json")
            with open(json_path, "w") as json_file:
                json.dump(settings, json_file)
            x_t.save(os.path.join(target_dir, str(i) + ".png"))
        

def read_all_pred_x_0():
    pred_x_0_list = []
    img_ls = ["edit_imgs", "ori_imgs"]
    root_dir = "/data/AI_paintings/imgs/"
    for i in range(2):
        target_dir = os.path.join(root_dir, img_ls[i])
        for root, dirs, files in os.walk(target_dir):
            if dirs and len(dirs[0]) == 14:
                for i in range(len(dirs)):
                    hashcode = dirs[i]
                    next_dir = os.path.join(target_dir, hashcode, "x_t")
                    for file in os.listdir(next_dir):
                        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                            img = Image.open(os.path.join(next_dir, file))
                            pred_x_0_list.append((img, hashcode))

    # return pred_x_0_list
    return sorted(pred_x_0_list, key=lambda x: x[1], reverse=True)
    
        
    

def read_file(hashcode: str):
    root_dir = "/data/AI_paintings/imgs/"
    img_ls = ["edit_imgs", "ori_imgs"]
    denoised_pred_list = []
    for index in range(2):
        target_dir = os.path.join(root_dir, img_ls[index], hashcode)
        if os.path.exists(target_dir):
            next_dir = os.path.join(target_dir, "denosed_prediction")
            count = 0
            for file in os.listdir(next_dir):
                img = Image.open(os.path.join(next_dir, str(count) + ".png"))
                count = count + 1
                denoised_pred_list.append(img)
            break
    return denoised_pred_list




