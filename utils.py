import os
import math
import numpy as np
import tensorflow as tf
from PIL import Image


def load_img(img_path, img_size=None):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.cast(img, dtype=tf.float32)
    
    if img_size:
        img = tf.image.resize(
            img,
            img_size,
            method=tf.image.ResizeMethod.BILINEAR,
            preserve_aspect_ratio=False,
            antialias=True
        )
    return img[tf.newaxis, ...]

def load_imgs(img_dir, img_size):
    imgs = []
    for dirpath, dirnames, filenames in os.walk(img_dir):
        for filename in filenames:
            img = load_img(os.path.join(dirpath, filename), img_size)
            imgs.append(img)
    imgs = tf.concat(imgs, axis=0)
    return imgs

def save_img(img, path):
    img = denormalize_m11(img)
    img = clip_0_255(img)
    img = Image.fromarray(img.numpy().astype(np.uint8).squeeze())
    img.save(path)

def imresize(img, min_size=0, scale_factor=None, new_shapes=None):
    assert not (scale_factor is None and new_shapes is None)
    if new_shapes is not None:
        new_height = new_shapes[1]
        new_width = new_shapes[2]
    elif scale_factor is not None:
        new_height = max(int(img.shape[1] * scale_factor), min_size)
        new_width = max(int(img.shape[2] * scale_factor), min_size)
    img = tf.image.resize(
        img,
        [new_height, new_width],
        method=tf.image.ResizeMethod.BILINEAR,
        antialias=True
    )
    return img

def create_pyramid(img, num_scales, scale_factor, min_size=0):
    pyramid = []
    for i in range(num_scales):
        pyramid.append(imresize(img, min_size, scale_factor=scale_factor ** i))
    pyramid.reverse()
    return pyramid

def get_conv_filters(scale):
    return 32 * (2 ** (scale // 4))

def normalize_m11(img):
    return img / 127.5 - 1.0

def denormalize_m11(img):
    return (img + 1.0) * 127.5

def clip_0_255(img):
     return tf.clip_by_value(img, clip_value_min=0.0, clip_value_max=255.0)
