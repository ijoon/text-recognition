import os
from random import shuffle

import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.python.ops.image_ops_impl import image_gradients


def text_to_labels(letters, text) -> list:
    return list(map(lambda x: letters.index(x), text))


def get_image_paths_and_string_labels(directory,
                                      letters,
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
                labels.append(text_to_labels(letters, label))
                image_paths.append(os.path.join(path, file_name))

    return image_paths, labels


def _map_path_to_tf_image(image_path, label, image_size_hw):
    # return dtype => tf.uint8
    image = tf.io.decode_image(tf.io.read_file(image_path),
        expand_animations=False)

    # return dtype => tf.float32
    image = tf.image.resize(image, image_size_hw)

    return image, label


def get_tf_dataset_for_images_and_string_labels(image_paths, labels, 
                                                image_size_hw):
    image_paths = tf.convert_to_tensor(image_paths, dtype=tf.string)
    labels = tf.convert_to_tensor(labels, dtype=tf.int64)

    ds = tf.data.Dataset.from_tensor_slices((image_paths, labels))

    ds = ds.map(
        lambda image_path, label: _map_path_to_tf_image(
            image_path=image_path,
            label=label,
            image_size_hw=image_size_hw),
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

@tf.function
def tf_random_condition():
    return tf.random.uniform([], 0, 1.0, dtype=tf.float32)

@tf.function
def tf_blur_image(x):
    def _gfilter(x):
        return tfa.image.gaussian_filter2d(x, [5, 5], 1.0, 'REFLECT', 0)

    def _mfilter(x):
        return tfa.image.median_filter2d(x, [5, 5], 'REFLECT', 0)

    return tf.cond(tf_random_condition() > 0.5,
        lambda: _gfilter(x), 
        lambda: _mfilter(x))

@tf.function
def tf_random_augment_image(x: tf.Tensor, p=0.2):

    x = tf.cond(tf_random_condition() < p, 
        lambda: tf.image.random_hue(x, 0.1), lambda: x)

    x = tf.cond(tf_random_condition() < 1, 
        lambda: tf.image.random_brightness(x, 0.5), lambda: x)

    x = tf.cond(tf_random_condition() < p, 
        lambda: tf.image.random_contrast(x, 0.9, 1.1), lambda: x)

    x = tf.cond(tf_random_condition() < p, 
        lambda: tf.image.random_saturation(x, 0.5, 1.5), lambda: x)

    x = tf.cond(tf_random_condition() < p, 
        lambda: tf.image.grayscale_to_rgb(tf.image.rgb_to_grayscale(x)),
        lambda: x)

    x = tf.cond(tf_random_condition() < p, 
        lambda: tf_blur_image(x), lambda: x)

    # if tf_random_condition() < p:
    #     x = tf.image.random_hue(x, 0.1)

    # if tf_random_condition() < p:
    #     x = tf.image.random_brightness(x, 0.1)

    # if tf_random_condition() < p:    
    #     x = tf.image.random_contrast(x, 0.9, 1.1)

    # if tf_random_condition() < p:
    #     x = tf.image.random_saturation(x, 0.5, 1.5)

    # if tf_random_condition() < p:
    #     x = tf_blur_image(x)

    # if tf_random_condition() < 0.1:
    #     x = tf.image.grayscale_to_rgb(tf.image.rgb_to_grayscale(x))

    return x

@tf.function
def normalize_image(x: tf.Tensor):
    # Normalize images to the range [0, 1].
    return x / 255.

@tf.function
def _map_preprocess_data(image, label, augmentation=False, normalization=True):    
    if augmentation:
        image = tf_random_augment_image(image)

    if normalization:
        image = normalize_image(image)

    return image, label



def _map_batch_for_ctc_loss(images, labels, image_w, downsample_factor, 
                            batch_size, text_length, 
                            augmentation, normalization):

    if augmentation:
        images = tf.map_fn(fn=tf_random_augment_image, elems=images,
            parallel_iterations=batch_size)

    if normalization:
        images = normalize_image(images)


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
        train_ds => shuffle
    5. 배치, preprocess, ctc_loss
    6. prefetch

    """
    
    image_paths, labels = get_image_paths_and_string_labels(
        directory='data_generate/generated_imgs',
        allow_image_formats=('.jpeg', '.jpg'),
        letters='0123456789-. ',
        label_length=5)
        
