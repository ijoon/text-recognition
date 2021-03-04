import os

import cv2
import numpy as np


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
                    unit_size: set or list,
                    seq_length: int,
                    align_left=True):
    units = list(unit_chars.keys())
    imgs = []
    random_seq_len = np.random.randint(seq_length, seq_length+1)
    label = ""
    for rnd_pick in np.random.randint(0, len(units), random_seq_len):
        img = np.array(unit_chars[units[rnd_pick]][0])
        img = fit_to_size(img, unit_size)
        imgs.append(img)
        label += units[rnd_pick]
    if not align_left:
        label = label[::-1]
    img = stitch_chars(imgs, (unit_size[0] * random_seq_len, unit_size[1]), align_left)
    return img, label

def augment_background(img, bg_img, unit_size, min_length):
    scale = np.random.rand() / 2 + 0.5
    rows, cols = img.shape[:2]
    bg_cols = unit_size[0] * min_length
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

if __name__ == "__main__":
    path = 'base_data'
    unit_files = {}
    for dir_name in os.listdir(path):
        if dir_name not in TARGET_CHARS:
            continue

        if not os.path.isdir(os.path.join(path, dir_name)):
            continue

        unit_files[dir_name] = []
        for unit_file in os.listdir(os.path.join(path, dir_name)):
            if '.jpg' not in unit_file:
                continue
            img = cv2.imread(os.path.join(path, dir_name, unit_file))
            unit_files[dir_name].append(img)

    print('# of unit = {}'.format(len(unit_files)))
    bg_img = cv2.imread('background.jpg')

    for n in range(5000):
        align = np.random.randint(0, 2)
        seq_img, label = combinate_chars(unit_files, (32, 64), 3, align)
        seq_img = augment_background(seq_img, bg_img, (32, 64), 3)
        print(label)
        seq_img = shear_img_randomly(seq_img)
        # cv2.imshow('img', seq_img)
        # cv2.waitKey()

        target_folder = f'results/{label}'
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)
            count = 0
        else:
            count = len([f for f in os.listdir(target_folder) if '.jpg' in f])


        cv2.imwrite(f'{target_folder}/{label}_{count}.jpg', seq_img)
