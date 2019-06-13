
import argparse
import os
import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from lib.utils import *
from lib.ops import discriminator_loss, generator_loss, l1_loss
from lib.datasets_mask import ImageDataset
from PIL import Image
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--n_iters', type=int, default=100000, help='number of iteration of training')
parser.add_argument('--batch_size', type=int, default=4, help='size of the batches')
parser.add_argument('--batch_size_val', type=int, default=8, help='size of the batches for validation')
parser.add_argument('--gpu_id', type=int, default=0, help='gpu_id')
parser.add_argument('--dataset_name', type=str, default='celebA', help='name of the dataset')
parser.add_argument('--crop_mode', type=str, default='none', help='[none|random]')
parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--n_cpu', type=int, default=4, help='number of cpu threads to use during batch generation')
parser.add_argument('--latent_dim', type=int, default=100, help='dimensionality of the latent space')
parser.add_argument('--img_size', type=int, default=128, help='size of each image dimension')
parser.add_argument('--load_size', type=int, default=280, help='size of each image loading dimension')
parser.add_argument('--mask_size', type=int, default=64, help='size of random mask')
parser.add_argument('--channels', type=int, default=3, help='number of image channels')
parser.add_argument('--sample_interval', type=int, default=500, help='interval between image sampling')
parser.add_argument('--n_rows', type=int, default=5, help='number of rows to generate for visualization')
parser.add_argument('--snapshot_interval', type=int, default=1, help='interval between image sampling')
parser.add_argument('--vis_interval', type=int, default=500, help='interval between image sampling')
parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='location to checkpoint')
parser.add_argument('--noise_w', type=float, default=5, help='weights for diversity-encouraging term')
parser.add_argument('--feat_w', type=float, default=10, help='weights for pixel-wise similarity term')
parser.add_argument('--noise_dim', type=int, default=32, help='dimmension for noise vector')
parser.add_argument('--no_noise', action='store_true', help='do not use the noise if specified')
parser.add_argument('--dist_measure', type=str, default='perceptual', help='[rgb | perceptual]')
parser.add_argument('--n_layers_G', type=int, default=3, help='number of layers in generator')
parser.add_argument('--n_layers_D', type=int, default=3, help='number of layers in discriminator')
parser.add_argument('--num_D', type=int, default=2, help='number of discriminators for multiscale PatchGAN')
args = parser.parse_args()
print(args)

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
tf.reset_default_graph()

# visualization
exp_name = 'CENoise_noiseDim_%d_lambda_%.3f_outputDist%s' % (args.noise_dim, args.noise_w, args.dist_measure)
if not args.crop_mode == 'random':
    exp_name += '_' + args.crop_mode
checkpoint_dir = os.path.abspath(os.path.join(args.checkpoint_dir, exp_name))
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
    
#x = os.path.normpath(checkpoint_dir).split(os.sep)
#if 'users' in x:
#    data_dir = '/home/users/ntu/yasin001/scratch/data/'
#elif 'yazici' in x:
#    data_dir = '/home/yazici/Documents/data/'
#else:
#    raise NotImplementedError("not implemented")
    
data_dir = '/home/yazici/Documents/data/' 
data_path = os.path.abspath(data_dir+"%s" % args.dataset_name)

# Initialize generator and discriminator
from models.Generator_NET import GlobalTwoStreamGenerator
generator = GlobalTwoStreamGenerator(output_nc=args.channels, z_dim = args.noise_dim, n_downsampling=args.n_layers_G, name="generator")
from models.Discriminator_NET import MultiscaleDiscriminator
discriminator = MultiscaleDiscriminator(n_layers=args.n_layers_D, num_D=args.num_D, name="discriminator")

# Dataset loader
dataloader = ImageDataset(data_path, img_size=args.img_size, load_size=args.load_size, 
                          mask_size=args.mask_size, crop_mode=args.crop_mode)
dataloader_test = ImageDataset(data_path, img_size=args.img_size, load_size=args.load_size, 
                               mask_size=args.mask_size, mode='val')

def data_manager(list_, indexes):
    img_list = []
    missing_list = []
    aux_list = []
    mask_list = []
    for i in indexes:
        tuple_ = list_[i]
        img_list.append(np.expand_dims(tuple_[0],0))
        missing_list.append(np.expand_dims(tuple_[1],0))
        aux_list.append(np.expand_dims(tuple_[2],0))
        mask_list.append(np.expand_dims(tuple_[3],0))
    
    return np.concatenate(img_list), np.concatenate(missing_list), np.concatenate(aux_list), np.concatenate(mask_list)

dataset = tf.data.Dataset.from_tensor_slices(range(args.n_iters * args.batch_size)).repeat()
iterator = dataset.batch(args.batch_size).prefetch(1).make_one_shot_iterator()
sample = iterator.get_next()

dataset = tf.data.Dataset.from_tensor_slices(range(args.n_iters * args.batch_size)).repeat()
iterator = dataset.batch(args.batch_size_val).prefetch(1).make_one_shot_iterator()
sample_val = iterator.get_next()

# data = data_manager(dataloader, sess.run(sample))
# data_val = data_manager(dataloader_test, sess.run(sample_val))

def sample_z(batch_size = args.batch_size):
    return np.random.normal(size=(batch_size, args.noise_dim)).astype(dtype='float32')

is_training_pl = tf.placeholder(tf.bool, [], name='is_training_pl')
img_ph = tf.placeholder(tf.float32, (None, args.img_size, args.img_size, args.channels), name='img_ph')
masked_img_ph = tf.placeholder(tf.float32, (None, args.img_size, args.img_size, args.channels), name='masked_img_ph')
noise_ph = tf.placeholder(tf.float32, (None, args.noise_dim), name='noise_ph')
mask_ph = tf.placeholder(tf.float32, (None, args.img_size, args.img_size, 1), name='mask_ph')        
        
#generator
gen_samples = generator.forward(masked_img_ph, noise_ph, mask_ph)
#discriminators
pred_real = discriminator.forward(img_ph)
pred_fake = discriminator.forward(gen_samples, reuse=True)

# loss for the discriminator
d_loss = []
if isinstance(pred_real[0], list):
    for i in range(len(pred_real)):
        d_loss.append(discriminator_loss('lsgan', pred_real[i][-1], pred_fake[i][-1]))
else:
    for i in range(pred_real):
        d_loss.append(discriminator_loss('lsgan', pred_real[i], pred_fake[i])) 
d_loss = tf.reduce_sum(d_loss)     

# adversiral loss for the generator        
g_loss = []
if isinstance(pred_fake[0], list):
    for pred in pred_fake:
        g_loss.append(generator_loss('lsgan', pred[-1]))
else:
    for pred in pred_fake:
        g_loss.append(generator_loss('lsgan', pred))

# pixel-wise loss for the generator
feat_weights = 4.0 / (args.n_layers_D)
D_weights = 1.0 / args.num_D
for i in range(args.num_D):
    for j in range(len(pred_fake[i])-1):
        g_loss.append(D_weights * feat_weights * args.feat_w * l1_loss(pred_fake[i][j], pred_real[i][j]))

# noise sensitivity loss
if args.dist_measure == 'rgb':
    gen_samples_1, gen_samples_2 = tf.split(gen_samples,2,axis=0)
    g_noise_out_dist = tf.reduce_mean(tf.abs(gen_samples_1 - gen_samples_2),axis=(1,2,3))
elif args.dist_measure == 'perceptual':
    g_noise_out_dist = 0
    for i in range(args.num_D):
        for j in range(len(pred_fake[i])-1):
            pred_fake_1, pred_fake_2 = tf.split(pred_fake[i][j],2,axis=0)
            g_noise_out_dist += D_weights * feat_weights * tf.reduce_mean(tf.abs(pred_fake_1 - pred_fake_2),axis=(1,2,3))

noise_1, noise_2 = tf.split(noise_ph,2,axis=0)
g_noise_z_dist = tf.reduce_mean(tf.abs(noise_1 - noise_2),axis=1)
g_noise = tf.reduce_mean( g_noise_out_dist/ g_noise_z_dist ) * args.noise_w
g_loss.append(-g_noise)

g_loss_list = g_loss
g_loss = tf.reduce_sum(g_loss) 

d_vars = tf.trainable_variables(scope="discriminator")
g_vars = tf.trainable_variables(scope="generator") 

#optimizers 
g_train_opt = tf.train.AdamOptimizer(args.lr,args.b1,args.b2).minimize(g_loss, var_list=g_vars)
d_train_opt = tf.train.AdamOptimizer(args.lr,args.b1,args.b2).minimize(d_loss, var_list=d_vars)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
        
fixed_sample = sess.run(sample_val)
for i in tqdm(range(args.n_iters+1)):
    
    data = data_manager(dataloader, sess.run(sample))
    n = sample_z(2*args.batch_size)
    #n = np.vstack([n]*2)
    image = np.vstack([data[0]]*2)
    masked_image = np.vstack([data[1]]*2)
    mask = np.vstack([data[-1]]*2)

    dl, _ = sess.run([d_loss, d_train_opt],{img_ph:image, masked_img_ph:masked_image, mask_ph:mask, noise_ph:n})         
    gl, _ = sess.run([g_loss, g_train_opt],{img_ph:image, masked_img_ph:masked_image, mask_ph:mask, noise_ph:n})
    
    if ((i) % args.vis_interval == 0):# and (i !=0):
        print("step: [%d] dl: [%.4f], gl: [%.4f]" %(i, dl, gl))
        print(sess.run(g_loss_list,{img_ph:image, masked_img_ph:masked_image, mask_ph:mask, noise_ph:n}))
    
    if ((i) % args.sample_interval == 0):# and (i !=0):
        data_val = data_manager(dataloader_test, fixed_sample)
        n = sample_z(args.n_rows)
        n = np.vstack([np.vstack([item]*args.batch_size_val) for item in n])
        masked_image = np.vstack([data_val[1]]*args.n_rows)
        mask = np.vstack([data_val[-1]]*args.n_rows)
        gen_img = sess.run(gen_samples,{masked_img_ph:masked_image, mask_ph:mask, noise_ph:n})
        gen_img_grid = make_grid(data_val[1], gen_img, args.n_rows)
        plt.imsave(checkpoint_dir+'/interpolation_{0:06}.jpg'.format(i), gen_img_grid)
        
