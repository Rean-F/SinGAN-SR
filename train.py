import os
from typing import List

import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.optimizers import Optimizer
from tensorflow.keras.models import Model
from tensorflow.keras.metrics import Mean
from tensorflow.keras.optimizers.schedules import ExponentialDecay

from model import Discriminator, Generator
from utils import imresize, create_pyramid, get_conv_filters, load_img, normalize_m11, save_img


class Trainer:
    def __init__(self, **kwargs) -> None:
        self.num_scales = kwargs["num_scale"]
        self.num_iters = kwargs["num_iters"]
        # num_filters double for every 4 scales
        self.num_filters = [get_conv_filters(scale) for scale in range(self.num_scales)]
        self.max_size = kwargs["max_size"]
        self.min_size = kwargs["min_size"]
        self.scale_factor = kwargs["scale_factor"]
        self.noise_amp_init = 0.1

        self.checkpoint_dir = kwargs["checkpoint_dir"]
        self.result_dir = kwargs["result_dir"]

        # total 3 * num_iters steps
        self.learning_schedule = ExponentialDecay(kwargs["learning_rate"], decay_steps=3*self.num_iters, decay_rate=0.1, staircase=True)
        self.build_model()

        self.debug = kwargs["debug"]
        if self.debug:
            self.create_summary_writer()
        return

    def build_model(self):
        # build a dis and gen for every scale
        self.discriminators = []
        self.generators = []

        for scale in range(self.num_scales):
            self.discriminators.append(Discriminator(self.num_filters[scale]))
            self.generators.append(Generator(self.num_filters[scale]))
        return
    
    def save_model(self, scale):
        scale_dir = os.path.join(self.checkpoint_dir, f"{scale}")
        if not os.path.exists(scale_dir):
            os.mkdir(scale_dir)
        G_file = os.path.join(scale_dir, "G/G")
        D_file = os.path.join(scale_dir, "D/D")
        self.generators[scale].save_weights(G_file, save_format="tf")
        self.discriminators[scale].save_weights(D_file, save_format="tf")
        np.save(self.checkpoint_dir + "/noise_amp.npy", self.noise_amps)

    def load_weights(self, scale):
        if self.num_filters[scale] == self.num_filters[scale - 1]:
            prev_scale_dir = os.path.join(self.checkpoint_dir, f"{scale - 1}")
            prev_G_file = os.path.join(prev_scale_dir, "G/G")
            prev_D_file = os.path.join(prev_scale_dir, "D/D")
            self.generators[scale].load_weights(prev_G_file)
            self.discriminators[scale].load_weights(prev_D_file)

    def train(self, img_path):
        real_img = load_img(img_path)
        real_img = normalize_m11(real_img)
        reals = create_pyramid(real_img, self.num_scales, self.scale_factor, self.min_size)

        for scale in range(self.num_scales):
            save_img(reals[scale], os.path.join(self.result_dir, f"pyramid_{scale}.png"))

        self.noise_amps = []

        for scale in range(self.num_scales):
            print(f"train scale {scale}")
            print(self.noise_amps)
            if scale > 0:
                self.load_weights(scale)
            
            real = reals[scale]
            discriminator = self.discriminators[scale]
            generator = self.generators[scale]
            d_opt = optimizers.Adam(learning_rate=self.learning_schedule, beta_1=0.5, beta_2=0.999)
            g_opt = optimizers.Adam(learning_rate=self.learning_schedule, beta_1=0.5, beta_2=0.999)
            prev_rec = self.generate_from_coarsest_rec(reals, scale)
            rmse = tf.sqrt(tf.reduce_mean(tf.square(real - prev_rec)))
            noise_amp = 1.0 if scale == 0 else self.noise_amp_init * rmse.numpy()
            for step in range(self.num_iters):
                metrics = self.train_step(reals, scale, step, prev_rec, noise_amp, discriminator, generator, d_opt, g_opt)
                if self.debug:
                    self.write_summary(scale, step, metrics)
            self.noise_amps.append(noise_amp)
            self.save_model(scale)
        return

    def train_step(
            self, 
            reals: List[tf.Tensor], scale: int, step: int,
            prev_rec: tf.Tensor, noise_amp: float, 
            discriminator: Model, generator: Model, 
            d_opt: Optimizer, g_opt: Optimizer):
        real = reals[scale]
        for i in range(6):
            prev_rand = self.generate_from_coarsest_rand(reals, scale)
            z_rand = tf.random.normal(real.shape) * noise_amp
            z_rec = tf.random.normal(real.shape) if scale == 0 else tf.zeros_like(real)
            # fitting discriminator
            if i < 3:
                with tf.GradientTape() as tape:
                    fake_rand = generator(prev_rand, z_rand)
                    dis_loss = self.discriminator_wgan_loss(discriminator, real, fake_rand)
                dis_gradients = tape.gradient(dis_loss, discriminator.trainable_variables)
                d_opt.apply_gradients(zip(dis_gradients, discriminator.trainable_variables))
            else:
                with tf.GradientTape() as tape:
                    fake_rand = generator(prev_rand, z_rand)
                    fake_rec= generator(prev_rec, z_rec)
                    gen_loss = self.generator_wgan_loss(discriminator, fake_rand)
                    rec_loss = self.generator_rec_loss(real, fake_rec)
                    gen_loss = gen_loss + 100.0 * rec_loss
                gen_gradients = tape.gradient(gen_loss, generator.trainable_variables)
                g_opt.apply_gradients(zip(gen_gradients, generator.trainable_variables))
        if step % 500 == 0:
            prev_rand = self.generate_from_coarsest_rand(reals, scale)
            z_rand = noise_amp * tf.random.normal(real.shape, dtype=tf.float32)
            fake = generator(prev_rand, z_rand)[0:1, ...]
            save_img(fake, os.path.join(self.result_dir, f"training_{scale}_{step}.png"))

        metrics = [dis_loss, gen_loss, rec_loss]
        return metrics

    def generate_from_coarsest_rand(self, reals, scale):
        fake = tf.zeros_like(reals[0])
        for i in range(scale):
            z_rand = tf.random.normal(reals[i].shape)
            z_rand = self.noise_amps[i] * z_rand
            fake = self.generators[i](fake, z_rand)
            fake = imresize(fake, new_shapes=reals[i+1].shape)
        return fake

    def generate_from_coarsest_rec(self, reals, scale):
        fake = tf.zeros_like(reals[0])
        for i in range(scale):
            if i == 0:
                z_rec = tf.random.normal(reals[i].shape, dtype=tf.float32)
            else:
                z_rec = tf.zeros_like(reals[i], dtype=tf.float32)
            fake = self.generators[i](fake, z_rec)
            fake = imresize(fake, new_shapes=reals[i+1].shape)
        return fake

    def discriminator_wgan_loss(self, discriminator: Model, real: tf.Tensor, fake: tf.Tensor):
        batch_size = real.shape[0]
        dis_loss = tf.reduce_mean(discriminator(fake)) - tf.reduce_mean(discriminator(real))
        alpha = tf.random.uniform(shape=[batch_size, 1, 1, 1], minval=0.0, maxval=1.0)
        interpolates = alpha * real + (1 - alpha) * fake
        with tf.GradientTape() as tape:
            tape.watch(interpolates)
            dis_interpolates = discriminator(interpolates)
        gradients = tape.gradient(dis_interpolates, [interpolates])[0]

        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=3))
        gradient_penalty = tf.reduce_mean(tf.square(slopes - 1.0))

        dis_loss = dis_loss + 0.1 * gradient_penalty
        return dis_loss

    def generator_wgan_loss(self, discriminator: Model, fake_rand: tf.Tensor):
        return -tf.reduce_mean(discriminator(fake_rand))

    def generator_rec_loss(self, real, fake_rec):
        return tf.reduce_mean(tf.square(real - fake_rec))

    def create_summary_writer(self):
        import datetime
        self.summary_writer = tf.summary.create_file_writer(
            "log/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        )

    def write_summary(self, scale, step, metrics):
        dis_loss, gen_loss, rec_loss = metrics
        with self.summary_writer.as_default():
            tf.summary.scalar(f"dis_loss_{scale}", dis_loss, step=step)
            tf.summary.scalar(f"gen_loss_{scale}", gen_loss, step=step)
            tf.summary.scalar(f"rec_loss_{scale}", rec_loss, step=step)
        # print("dis_loss\tgen_loss\trec_loss")
        # print(f"{dis_loss}\t{gen_loss}\t{rec_loss}")

    def create_metrics(self):
        self.dis_metric = Mean()
        self.gen_metric = Mean()
        self.rec_metric = Mean()

    def update_metrics(self, scale, metrics):
        dis_loss, gen_loss, rec_loss = metrics
        self.dis_metric(dis_loss)
        self.gen_metric(gen_loss)
        self.rec_metric(rec_loss)

        print(f"dis_loss = {self.dis_metric.result():.3f}")
        print(f"gen_loss = {self.gen_metric.result():.3f}")
        print(f"rec_loss = {self.rec_metric.result():.3f}")

        self.dis_metric.reset_states()
        self.gen_metric.reset_states()
        self.rec_metric.reset_states()


if __name__ == "__main__":
    import os

    img_name = "island"
    img_format = "png"
    img_path = os.path.join("dataset", img_name + "." + img_format)
    checkpoint_dir = os.path.join("models", img_name)
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    result_dir = os.path.join("data", img_name)
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    
    trainer = Trainer(
        num_scale=10,
        num_iters=2001,
        learning_rate=5e-4,
        max_size=960,
        min_size=12,
        scale_factor=0.75,
        checkpoint_dir=checkpoint_dir,
        result_dir=result_dir,
        debug=True
    )
    
    trainer.train(img_path)