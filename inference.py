import os
from typing import List

import numpy as np
import tensorflow as tf
from model import Generator
from utils import get_conv_filters, imresize, create_pyramid, load_img, normalize_m11, save_img


class Inferencer:
    def __init__(
            self, 
            num_samples, 
            num_scales,
            scale_factor, 
            min_size, 
            checkpoint_dir,
            result_dir) -> None:
        self.generators = []
        self.noise_amp = []
        self.num_samples = num_samples
        self.num_scales = num_scales
        self.scale_factor = scale_factor
        self.min_size = min_size
        self.result_dir = result_dir

        self.load_model(checkpoint_dir)

    def load_model(self, checkpoint_dir):
        self.noise_amp = np.load(checkpoint_dir + "/noise_amp.npy")
        for scale in range(self.num_scales):
            generator = Generator(num_filters=get_conv_filters(scale))
            generator.load_weights(os.path.join(checkpoint_dir, f"{scale}", "G/G"))
            self.generators.append(generator)
        return

    def inference_random(self, img_fname, inject_scale):
        real_img = load_img(img_fname)
        real_img = normalize_m11(real_img)
        reals = create_pyramid(real_img, self.num_scales, self.scale_factor, self.min_size)
        for i in range(self.num_samples):
            fake = self.random_generate(reals, inject_scale)
            save_img(fake, os.path.join(self.result_dir, f"random_{i}.png"))
    
    def random_generate(self, reals, inject_scale=0):
        if inject_scale > 0:
            inject_real = reals[inject_scale-1]
            fake = inject_real
        else:
            fake = tf.zeros_like(reals[0], dtype=tf.float32)
        for scale in range(inject_scale, self.num_scales):
            generator = self.generators[scale]
            fake = imresize(fake, new_shapes=reals[scale].shape)
            z = tf.random.normal(reals[scale].shape, dtype=tf.float32)
            z = self.noise_amp[scale] * z
            fake = generator(fake, z)
        return fake

    def inference_sr(self, img_path, sr_scale):
        real_img = load_img(img_path)
        real_img = normalize_m11(real_img)
        
        batch_size = 1
        num_iter = 4
        hs = [int(real_img.shape[1] * sr_scale ** (i/num_iter)) for i in range(num_iter+1)]
        ws = [int(real_img.shape[2] * sr_scale ** (i/num_iter)) for i in range(num_iter+1)]
        sr_generator = self.generators[-1]
        prev = real_img
        for h, w in zip(hs, ws):
            prev = imresize(prev, new_shapes=[batch_size, h, w])
            z = tf.random.uniform(prev.shape, dtype=tf.float32)
            z = self.noise_amp[-1] * z
            prev = sr_generator(prev, z)
        save_img(prev, os.path.join(self.result_dir, f"{sr_scale}x.png"))
    

if __name__ == "__main__":
    img_name = "island"
    img_format = "png"
    img_path = os.path.join("dataset", img_name + "." + img_format)

    checkpoint_dir = os.path.join("models", img_name)
    result_dir = os.path.join("data", img_name)
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    inferencer = Inferencer(
        num_samples=20,
        num_scales=8,
        scale_factor=0.75,
        min_size=2,
        checkpoint_dir=checkpoint_dir,
        result_dir=result_dir
    )
    inferencer.inference_sr(img_path, sr_scale=4)