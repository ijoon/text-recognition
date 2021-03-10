import os
from random import shuffle

import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.python.ops.image_ops_impl import image_gradients

def get_image_paths_and_string_labels(directory,
                                      label_length,
                                      allow_image_formats=('.jpeg', '.jpg')):

    """ Get image paths and string labels in a directory.

    The directory structure must be like:
    ```
    directory/
    ...class_a/
    ......a_image_1.jpg
    ......a_image_2.jpg
    ...class_b/
    ......b_image_1.jpg
    ```

    Args:
        directory: Directory where the data is located.
        allow_image_formats: Allow list of file extensions. (e.g. '.jpg').
        label_length: String label length (e.g. 5 => return '123  ')

    Returns: 
        (image_paths, labels)

        - image_paths
            ['directory/class_a/a_image1.jpg,
             'directory/class_a/a_image2.jpg,
             'directory/class_b/b_image1.jpg]
        - labels
            [class_a, class_a, class_b]

    """
    image_paths = []
    labels = []
    for (path, _, files) in os.walk(directory):
        for file_name in files:
            if file_name.lower().endswith(allow_image_formats):
                label = os.path.basename(path)

                if not label_length:
                    labels.append(label)
                labels.append(label.ljust(label_length))
                image_paths.append(os.path.join(path, file_name))

    return image_paths, labels


def _map_path_to_tf_image(image_path, label, image_size_h_w):
    # return dtype => tf.uint8
    image = tf.io.decode_image(tf.io.read_file(image_path),
        expand_animations=False)

    # return dtype => tf.float32
    image = tf.image.resize(image, image_size_h_w)
    return image, label


def get_tf_dataset_for_images_and_string_labels(image_paths, labels, 
                                                image_size_hw):
    image_paths = tf.convert_to_tensor(image_paths, dtype=tf.string)
    labels = tf.convert_to_tensor(labels, dtype=tf.string)

    ds = tf.data.Dataset.from_tensor_slices((image_paths, labels))

    ds = ds.map(
        lambda image_path, label: _map_path_to_tf_image(
            image_path=image_path,
            label=label,
            image_size_h_w=image_size_hw),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)

    return ds


def split_train_valid_for_tf_dataset(total_ds, valid_ratio=0.2, 
                                     shuffle=True, cache=True):
    dataset_size = total_ds.cardinality().numpy()
    valid_size = int(dataset_size * valid_ratio)
    train_size = dataset_size - valid_size

    if shuffle:
        total_ds = total_ds.shuffle(dataset_size)

    train_ds = total_ds.take(train_size)
    valid_ds = total_ds.skip(train_size)

    if cache:
        train_ds = train_ds.cache()
        valid_ds = valid_ds.cache()

    return train_ds, valid_ds


def split_train_valid(x, y, valid_ratio=0.2, shuffle=True):
    return train_test_split(x, y, 
        test_size=valid_ratio,
        random_state=1,
        shuffle=True)


def tf_random_condition():
    return tf.random.uniform([], 0, 1.0, dtype=tf.float32)


def tf_blur_image(x):
    choice = tf.random.uniform([], 0, 1, dtype=tf.float32)
    def _gfilter(x):
        return tfa.image.gaussian_filter2d(x, [5, 5], 1.0, 'REFLECT', 0)

    def _mfilter(x):
        return tfa.image.median_filter2d(x, [5, 5], 'REFLECT', 0)

    return tf.cond(choice > 0.5,
        lambda: _gfilter(x), 
        lambda: _mfilter(x))


def tf_random_augment_image(x: tf.Tensor, p=0.2):

    if tf_random_condition() < p:
        x = tf.image.random_hue(x, 0.1)

    if tf_random_condition() < p:
        x = tf.image.random_brightness(x, 0.1)

    if tf_random_condition() < p:    
        x = tf.image.random_contrast(x, 0.9, 1.1)

    if tf_random_condition() < p:
        x = tf.image.random_saturation(x, 0.5, 1.5)

    if tf_random_condition() < p:
        x = tf_blur_image(x)

    if tf_random_condition() < 0.1:
        x = tf.image.grayscale_to_rgb(tf.image.rgb_to_grayscale(x))

    return x


def normalize_image(x: tf.Tensor):
    # Normalize images to the range [0, 1].
    return x / 255.


def _map_preprocess_data(image, label, augmentation=False, normalization=True):    
    if augmentation:
        image = tf_random_augment_image(image)

    if normalization:
        image = normalize_image(image)

    return image, label


def _map_batch_for_ctc_loss(images, labels, image_w, downsample_factor, 
                            batch_size, text_length):

    input_length = tf.ones((batch_size, 1)) * (image_w//downsample_factor - 2)
    label_length = tf.ones((batch_size, 1)) * text_length 

    input_dict = {
        'inputs': images,
        'labels': labels,
        'input_length': input_length,
        'label_length': label_length
    }
    output_dict = {'ctc': tf.zeros([batch_size])}

    return input_dict, output_dict

if __name__ == '__main__':
    
    """
    1. 이미지 경로 / 라벨 읽기
    2. 이미지 로드
    3. split / valid 나누기 (cache)
    4. epoch마다 
        train_ds => augmentation, normalize, shuffle
        valid_ds => normalize
    5. 배치, ctc_loss
    6. prefetch

    """
    
    image_paths, labels = get_image_paths_and_string_labels(
        directory='data_generate/results',
        allow_image_formats=('.jpeg', '.jpg'), 
        label_length=9)

    total_ds = get_tf_dataset_for_images_and_string_labels(
        image_paths=image_paths, 
        labels=labels, 
        image_size_hw=(64,128))

    train_ds, valid_ds = split_train_valid_for_tf_dataset(
        total_ds, 
        valid_ratio=0.2, 
        shuffle=True, 
        cache=True)
    
    train_ds = (
        train_ds
        .map(
            lambda image, label: _map_preprocess_data(
                image=image, 
                label=label, 
                augmentation=True, 
                normalization=True),
            num_parallel_calls=tf.data.experimental.AUTOTUNE)
        .shuffle(train_ds.cardinality().numpy())
        .batch(32)
        .map(
            lambda images, labels: _map_batch_for_ctc_loss(
                images=images,
                labels=labels,
                image_w=128,
                downsample_factor=4),
            num_parallel_calls=tf.data.experimental.AUTOTUNE)
        .prefetch(tf.data.experimental.AUTOTUNE)
    )

    valid_ds = (
        valid_ds
        .map(
            lambda image, label: _map_preprocess_data(
                image=image, 
                label=label, 
                augmentation=False, 
                normalization=True),
            num_parallel_calls=tf.data.experimental.AUTOTUNE)
        .shuffle(valid_ds.cardinality().numpy())
        .batch(32)
        .map(
            lambda images, labels: _map_batch_for_ctc_loss(
                images=images,
                labels=labels,
                image_w=128,
                downsample_factor=4),
            num_parallel_calls=tf.data.experimental.AUTOTUNE)
        .prefetch(tf.data.experimental.AUTOTUNE)
    )

