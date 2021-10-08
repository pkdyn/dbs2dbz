from __future__ import print_function, division
import scipy
import scipy.misc
from keras.datasets import mnist
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
#from tensorflow_addons.layers import InstanceNormalization
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import imageio
import numpy as np
import streamlit as st
from glob import glob
from PIL import Image

fav = Image.open("shenron.ico")
st.set_page_config(
    page_title="dbs2dbz: CycleGAN",
    page_icon = fav,
)

st.title("dbs2dbz")

class DataLoader():
    def __init__(self, dataset_name, img_res=(512, 512)):
        self.dataset_name = dataset_name
        self.img_res = img_res

    def load_data(self, domain, batch_size=1, is_testing=False):
        data_type = "train%s" % domain if not is_testing else "test%s" % domain
        path = glob('/content/%s/*' % ( data_type))

        batch_images = np.random.choice(path, size=batch_size)

        imgs = []
        for img_path in batch_images:
            img = self.imread(img_path)
            if not is_testing:
                img = scipy.misc.imresize(img, self.img_res)

                if np.random.random() > 0.5:
                    img = np.fliplr(img)
            else:
                img = scipy.misc.imresize(img, self.img_res)
            imgs.append(img)

        imgs = np.array(imgs)/127.5 - 1.

        return imgs

    def load_batch(self, batch_size=1, is_testing=False):
        data_type = "train" if not is_testing else "val"
        path_A = glob('/content/%sA/*' % ( data_type))
        path_B = glob('/content/%sB/*' % ( data_type))

        self.n_batches = int(min(len(path_A), len(path_B)) / batch_size)
        total_samples = self.n_batches * batch_size

        # Sample n_batches * batch_size from each path list so that model sees all
        # samples from both domains
        path_A = np.random.choice(path_A, total_samples, replace=False)
        path_B = np.random.choice(path_B, total_samples, replace=False)

        for i in range(self.n_batches-1):
            batch_A = path_A[i*batch_size:(i+1)*batch_size]
            batch_B = path_B[i*batch_size:(i+1)*batch_size]
            imgs_A, imgs_B = [], []
            for img_A, img_B in zip(batch_A, batch_B):
                img_A = self.imread(img_A)
                img_B = self.imread(img_B)

                img_A = scipy.misc.imresize(img_A, self.img_res)
                img_B = scipy.misc.imresize(img_B, self.img_res)

                if not is_testing and np.random.random() > 0.5:
                        img_A = np.fliplr(img_A)
                        img_B = np.fliplr(img_B)

                imgs_A.append(img_A)
                imgs_B.append(img_B)

            imgs_A = np.array(imgs_A)/127.5 - 1.
            imgs_B = np.array(imgs_B)/127.5 - 1.
            yield imgs_A, imgs_B

    def imread(self, path):
        return scipy.misc.imread(path, mode='RGB').astype(np.float)

class CycleGAN():
    def __init__(self):
        # Input shape
        self.img_rows = 512
        self.img_cols = 512
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        # Configure data loader
        self.dataset_name = 'dbs2dbz'
        # Use the DataLoader object to import a preprocessed dataset
        self.data_loader = DataLoader(dataset_name=self.dataset_name,
                                      img_res=(self.img_rows, self.img_cols))

        # Calculate output shape of D (PatchGAN)
        patch = int(self.img_rows / 2**4)
        self.disc_patch = (patch, patch, 1)
        # Number of filters in the first layer of G and D
        self.gf = 32
        self.df = 64
        # Loss weights
        self.lambda_cycle = 10.0                    # Cycle-consistency loss
        self.lambda_id = 0.9 * self.lambda_cycle    # Identity loss
        optimizer = Adam(0.0002, 0.5)
        # Build and compile the discriminators
        self.d_A = self.build_discriminator()
        self.d_B = self.build_discriminator()
        self.d_A.compile(loss='mse',
                         optimizer=optimizer,
                         metrics=['accuracy'])
        self.d_B.compile(loss='mse',
                         optimizer=optimizer,
                         metrics=['accuracy'])

        # Build the generators
        self.g_AB = self.build_generator()
        self.g_BA = self.build_generator()
        # Input images from both domains
        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)
        # Translate images to the other domain
        fake_B = self.g_AB(img_A)
        fake_A = self.g_BA(img_B)
        # Translate images back to original domain
        reconstr_A = self.g_BA(fake_B)
        reconstr_B = self.g_AB(fake_A)
        # Identity mapping of images
        img_A_id = self.g_BA(img_A)
        img_B_id = self.g_AB(img_B)
        # For the combined model we will only train the generators
        self.d_A.trainable = False
        self.d_B.trainable = False
        # Discriminators determines validity of translated images
        valid_A = self.d_A(fake_A)
        valid_B = self.d_B(fake_B)
        # Combined model trains generators to fool discriminators
        self.combined = Model(inputs=[img_A, img_B],
                              outputs=[valid_A, valid_B,
                                       reconstr_A, reconstr_B,
                                       img_A_id, img_B_id])
        self.combined.compile(loss=['mse', 'mse',
                                    'mae', 'mae',
                                    'mae', 'mae'],
                              loss_weights=[1, 1,
                                            self.lambda_cycle, self.lambda_cycle,
                                            self.lambda_id, self.lambda_id],
                              optimizer=optimizer)


class CycleGAN(CycleGAN):

    @staticmethod
    def conv2d(layer_input, filters, f_size=4, normalization=True):
        """Discriminator layer"""
        d = Conv2D(filters, kernel_size=f_size,
                    strides=2, padding='same')(layer_input)
        d = LeakyReLU(alpha=0.2)(d)
        if normalization:
            d = InstanceNormalization()(d)
        return d

    @staticmethod
    def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
        """Layers used during upsampling"""
        u = UpSampling2D(size=2)(layer_input)
        u = Conv2D(filters, kernel_size=f_size, strides=1,
                    padding='same', activation='relu')(u)
        if dropout_rate:
            u = Dropout(dropout_rate)(u)
        u = InstanceNormalization()(u)
        u = Concatenate()([u, skip_input])
        return u

    def build_generator(self):
        """U-Net Generator"""
            # Image input
        d0 = Input(shape=self.img_shape)
            # Downsampling
        d1 = self.conv2d(d0, self.gf)
        d2 = self.conv2d(d1, self.gf * 2)
        d3 = self.conv2d(d2, self.gf * 4)
        d4 = self.conv2d(d3, self.gf * 8)
            # Upsampling
        u1 = self.deconv2d(d4, d3, self.gf * 4)
        u2 = self.deconv2d(u1, d2, self.gf * 2)
        u3 = self.deconv2d(u2, d1, self.gf)
        u4 = UpSampling2D(size=2)(u3)
        output_img = Conv2D(self.channels, kernel_size=4,
                                strides=1, padding='same', activation='tanh')(u4)

        return Model(d0, output_img)

    def build_discriminator(self):
        img = Input(shape=self.img_shape)

        d1 = self.conv2d(img, self.df, normalization=False)
        d2 = self.conv2d(d1, self.df * 2)
        d3 = self.conv2d(d2, self.df * 4)
        d4 = self.conv2d(d3, self.df * 8)

        validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)
        return Model(img, validity)

    def sample_images(self, epoch, batch_i):
        r, c = 1, 2
        imgs_A = self.data_loader.load_data(domain="A", batch_size=1, is_testing=True)
        imgs_B = self.data_loader.load_data(domain="B", batch_size=1, is_testing=True)
        # Translate images to the other domain
        fake_B = self.g_AB.predict(imgs_A)
        fake_A = self.g_BA.predict(imgs_B)
        # Translate back to original domain
        reconstr_A = self.g_BA.predict(fake_B)
        reconstr_B = self.g_AB.predict(fake_A)
        #gen_imgs = np.concatenate([imgs_A, fake_B, reconstr_A, imgs_B, fake_A, reconstr_B])
        gen_imgs = np.concatenate([imgs_A, fake_B])
        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5
        titles = ['Original', 'Translated']
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[j].imshow(gen_imgs[cnt])
                axs[j].set_title(titles[j])
                axs[j].axis('off')
                cnt += 1
        fig.savefig("/content/%d_%d.png" % ( epoch, batch_i))
        plt.show()

    def train(self, epochs, batch_size=1, sample_interval=50):
        valid = np.ones((batch_size,) + self.disc_patch)
        fake = np.zeros((batch_size,) + self.disc_patch)
        for epoch in range(epochs):
            for batch_i, (imgs_A, imgs_B) in enumerate(self.data_loader.load_batch(batch_size)):
                # Translate images to opposite domain
                fake_B = self.g_AB.predict(imgs_A)
                fake_A = self.g_BA.predict(imgs_B)
                # Train the discriminators (original images = real / translated = Fake)
                dA_loss_real = self.d_A.train_on_batch(imgs_A, valid)
                dA_loss_fake = self.d_A.train_on_batch(fake_A, fake)
                dA_loss = 0.5 * np.add(dA_loss_real, dA_loss_fake)
                dB_loss_real = self.d_B.train_on_batch(imgs_B, valid)
                dB_loss_fake = self.d_B.train_on_batch(fake_B, fake)
                dB_loss = 0.5 * np.add(dB_loss_real, dB_loss_fake)
                # Total discriminator loss
                d_loss = 0.5 * np.add(dA_loss, dB_loss)
                # Train the generators
                g_loss = self.combined.train_on_batch([imgs_A, imgs_B],
                                                        [valid, valid, imgs_A, imgs_B, imgs_A, imgs_B])
                # If at save interval => plot the generated image samples
                if batch_i % sample_interval == 0:
                    self.sample_images(epoch, batch_i)

cycle_gan = CycleGAN()
#cycle_gan = pickle.load(open('dbz.pickle', 'rb'))
cycle_gan.g_AB.load_weights('gAB_s2z_43_8_100.h5')
uploaded_file = st.file_uploader("Choose an image...", type="jpg")
if uploaded_file is not None:
    uimg = Image.open(uploaded_file)
    uimg = uimg.save('im.jpg')
    #st.image(uimg, caption='Uploaded Image', use_column_width=True)
    if st.button('Transform'):
        img_res=(512, 512)
        imgs = []
        img = imageio.imread('im.jpg', pilmode = "RGB")
        img = np.array(Image.fromarray(img).resize(img_res))
        imgs.append(img)
        imgs = np.array(imgs)/127.5 - 1.
        imgs_A = imgs
        fake_B = cycle_gan.g_AB.predict(imgs_A)
        gen_imgs = np.concatenate([imgs_A, fake_B])
        gen_imgs = 0.5 * gen_imgs + 0.5
        
        #st.image(gen_imgs[0], use_column_width=True)
    
        r, c = 1, 2
        plt.style.use("dark_background")
        titles = ['Original', 'Z-style']
        fig, axs = plt.subplots(r, c)
        cnt = 0
         
        for i in range(r):
            for j in range(c):
                axs[j].imshow(gen_imgs[cnt], interpolation = 'hamming')
                axs[j].set_title(titles[j])
                axs[j].axis('off')
                cnt += 1
        #fig.savefig('dbz.jpg', dpi=600)
        #dimg = Image.open('dbz.jpg')
        #dimg.save('dbz.jpg', quality = 95)
        #st.image(dimg)
        #dim.show()
        st.pyplot(fig)
        st.image(gen_imgs[1], use_column_width=True)
    
    #os.remove('im.jpg')
