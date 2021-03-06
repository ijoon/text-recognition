import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from collections import defaultdict
import random

import cv2
import numpy as np

from config import cfg

TARGET_CHARS = '0123456789'

def temp(img):
    #Parameters of the affine transform:
    angle = 0#np.random.randint(0, 45)
    shear_x = np.random.rand()/2 - 0.25
    shear_y = np.random.rand()/2 - 0.25

    type_border = cv2.BORDER_CONSTANT
    color_border = (0, 0, 0)
    rows, cols = img.shape[:2]

    #First: Necessary space for the rotation
    M = cv2.getRotationMatrix2D((cols/2,rows/2), angle, 1)
    cos_part = np.abs(M[0, 0])
    sin_part = np.abs(M[0, 1])
    new_cols = int((rows * sin_part) + (cols * cos_part)) 
    new_rows = int((rows * cos_part) + (cols * sin_part))

    #Second: Necessary space for the shear
    new_cols += (abs(shear_x + shear_y)*new_cols)
    new_rows += (abs(shear_y + shear_x)*new_rows)

    #Calculate the space to add with border
    left_right = int((new_cols-cols)/2)
    up_down = int((new_rows-rows)/2)

    final_image = cv2.copyMakeBorder(img, up_down, up_down, left_right, left_right, type_border, value = color_border)
    rows,cols = final_image.shape[:2]

    translat_center_x = -(shear_x * cols) / 2
    translat_center_y = -(shear_y * rows) / 2
    M = M + np.float64([[0, shear_x, translat_center_x], [shear_y, 0, translat_center_y]])
    final_image  = cv2.warpAffine(final_image, M, (cols,rows), borderMode = type_border, borderValue = color_border)
    return final_image

def shear_img_randomly(img):
    xy = np.random.rand(3) / 2 + 0.5
    xs = np.random.rand() / 4 - 0.125
    ys = np.random.rand() / 4 - 0.125
    rows, cols = img.shape[:2]
    M = np.float32([[xy[0]    , 0, 0],
                    [ys, xy[1], 0],
                    [0    , 0    , 1]])
    cols = int(cols * xy[0])
    rows = int(rows * xy[1] + rows * ys * 2)
    sheared_img = cv2.warpPerspective(img, M,(cols, rows))
    return sheared_img

def stitch_chars(imgs, canvas_size, align_left=True):
    canvas_size = [canvas_size[1], canvas_size[0], 3]
    canvas = np.zeros(canvas_size, dtype=np.uint8)

    loc = 0 if align_left else canvas_size[1]
    for im in imgs:
        if align_left:
            canvas[:, loc:loc+im.shape[1], :] = im
            loc += im.shape[1]
        else:
            canvas[:, loc-im.shape[1]:loc, :] = im
            loc -= im.shape[1]

    return canvas


def fit_to_size(img, size):
    """
    The img is copied into center of canvas.
    """
    ratio = img.shape[0] / size[1]
    cwidth = int(np.ceil(img.shape[1] / ratio + 0.5))
    cheight = int(np.ceil(img.shape[0] / ratio + 0.5))
    cwidth = np.minimum(cwidth, size[0])
    cheight = np.minimum(cheight, size[1])
    target_shape = (cwidth, cheight)
    img = cv2.resize(img, target_shape)
    canvas = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    sx = (size[0] - img.shape[1]) // 2
    sy = (size[1] - img.shape[0]) // 2
    canvas[sy:sy+img.shape[0], sx:sx+img.shape[1], :] = img
    return canvas

def combinate_chars(unit_chars: dict,
                    unit_size_wh: set or list,
                    seq_length: int,
                    align_left=True):
    units = list(unit_chars.keys())
    imgs = []
    label = ''
    for rnd_pick in np.random.randint(0, len(units), seq_length):
        img = random.choice(unit_chars[units[rnd_pick]])
        img = fit_to_size(img, unit_size_wh)
        imgs.append(img)
        label += units[rnd_pick]
    if not align_left:
        label = label[::-1]
    img = stitch_chars(imgs, (unit_size_wh[0] * seq_length, unit_size_wh[1]), 
        align_left)

    return img, label

def augment_background(img, bg_img, unit_size_wh, max_length):
    scale = np.random.rand() / 2 + 0.5
    rows, cols = img.shape[:2]
    bg_cols = unit_size_wh[0] * max_length
    range_x = bg_img.shape[1] - bg_cols
    range_y = bg_img.shape[0] - rows
    crop_x = np.random.randint(0, range_x)
    crop_y = np.random.randint(0, range_y)
    crop_w = bg_cols
    crop_h = rows
    crop_bg = np.array(bg_img[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w, :])
    resize_shape = (int(cols*scale), int(rows*scale))
    img = cv2.resize(img, resize_shape)
    rows, cols = img.shape[:2]
    m_top = np.random.randint(0, crop_h-rows) if scale != 1.0 else 0
    m_left = np.random.randint(0, crop_w-cols) if scale != 1.0 else 0
    crop_bg[m_top:m_top+rows, m_left:m_left+cols, :] = img[:, :, :]
    return crop_bg


def augment_background2(seq_img, 
                        label, 
                        bg_img, 
                        unit_size_wh, 
                        max_length, 
                        blank_left):
    """
    1. max_length만큼 background 영역 crop (random margin)
    2. blank left를 통해 seq_img를 왼쪽으로 붙일지 오른쪽으로 붙일지 결정 (label도 변경)
    3. 결과 image, label 리턴
    """

    bg_w = unit_size_wh[0] * max_length
    bg_h = unit_size_wh[1]
    
    l_margin = round(bg_w * random.uniform(0.0, 0.1))
    r_margin = round(bg_w * random.uniform(0.0, 0.1))
    t_margin = round(bg_h * random.uniform(0.0, 0.1))
    b_margin = round(bg_h * random.uniform(0.0, 0.1))

    total_bg_w = round(bg_w + l_margin + r_margin)
    total_bg_h = round(bg_h + t_margin + b_margin)

    if bg_img.shape[1] < total_bg_w or bg_img.shape[0] < total_bg_h:
        raise ValueError('Background image is too small.')

    bg_start_w = random.randint(0, bg_img.shape[1] - total_bg_w)
    bg_start_h = random.randint(0, bg_img.shape[0] - total_bg_h)

    crop_bg = bg_img[bg_start_h:bg_start_h+total_bg_h,
                     bg_start_w:bg_start_w+total_bg_w, :].copy()

    seq_h = seq_img.shape[0]
    seq_w = seq_img.shape[1]
    if blank_left:
        crop_h = crop_bg.shape[0]
        crop_w = crop_bg.shape[1]
        crop_bg[crop_h-b_margin-seq_h:crop_h-b_margin,
                crop_w-r_margin-seq_w:crop_w-r_margin, :] = seq_img[:,:,:]

        label = label.rjust(max_length)
    else:
        crop_bg[t_margin:t_margin+seq_h, 
                l_margin:l_margin+seq_w, :] = seq_img[:,:,:]
        label = label.ljust(max_length)
        
    return crop_bg, label
    

def generate_multi_digit_imgs(base_data_dir,
                              bg_data_dir,
                              min_length,
                              max_length,
                              unit_img_size_wh,
                              data_size,
                              target_dir):

    # get unit imgs
    unit_img_dict = {}
    for unit_dir in os.listdir(base_data_dir):
        if unit_dir not in TARGET_CHARS:
            continue

        if not os.path.isdir(os.path.join(base_data_dir, unit_dir)):
            continue

        unit_img_dict[unit_dir] = []
        for unit_file in os.listdir(os.path.join(base_data_dir, unit_dir)):
            if not unit_file.lower().endswith(('.jpeg', '.jpg', '.png')):
                continue

            # Some mac file names start with '._'
            if unit_file.lower().startswith(('._')):
                continue

            img = cv2.imread(os.path.join(base_data_dir, unit_dir, unit_file))
            unit_img_dict[unit_dir].append(img)

    # get bg imgs
    bg_imgs = []
    for bg_file in os.listdir(bg_data_dir):
        if not bg_file.lower().endswith(('.jpeg', '.jpg', '.png')):
            continue

        # Some mac file names start with '._'
        if bg_file.lower().startswith(('._')):
            continue

        img = cv2.imread(os.path.join(bg_data_dir, bg_file))
        bg_imgs.append(img)

    # prob for random seq length 
    # length_list = list(range(min_length, max_length+1))
    # p = [pow(10,l) for l in length_list]
    # sum_p = sum(p)
    # p = [a / sum_p for a in p]

    length_list = list(range(min_length, max_length+1))
    sum_length_list = sum(length_list)
    p = [a / sum_length_list for a in length_list]

    # generate random seq
    for _ in range(data_size):
        seq_length = np.random.choice(length_list, p=p)
        seq_img, label = combinate_chars(
            unit_chars=unit_img_dict,
            unit_size_wh=unit_img_size_wh,
            seq_length=seq_length,
            align_left=random.choice([True, False]))

        seq_img, label = augment_background2(
            seq_img=seq_img, 
            label=label, 
            bg_img=random.choice(bg_imgs),
            unit_size_wh=unit_img_size_wh,
            max_length=max_length,
            blank_left=random.choice([True, False]))

        target_label_dir= f'{target_dir}/{label}'
        if not os.path.exists(target_label_dir):
            os.makedirs(target_label_dir)
            count = 0
        else:
            count = len(
                [f for f in os.listdir(target_label_dir) if '.jpg' in f])

        cv2.imwrite(f'{target_label_dir}/{label}_{count}.jpg', seq_img)


if __name__ == '__main__':

    # img_path = '/Users/rudy/Desktop/test.png'
    # img = cv2.imread(img_path)
    # cv2.imshow('test1', img)
    # fit_img = fit_to_size(img, (32,64))
    # cv2.imshow('test2', fit_img)
    # cv2.waitKey()

    generate_multi_digit_imgs(
        base_data_dir='data_generate/base_data',
        bg_data_dir='data_generate/bg_data',
        min_length=1,
        max_length=cfg['max_text_len'],
        unit_img_size_wh=(32, 64),
        data_size=1000,
        target_dir='data_generate/generated_imgs'
    )
