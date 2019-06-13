import tensorflow as tf
from lib.ops import * 
import numpy as np

##############################################################################
# Multi-Scale Discriminator
##############################################################################
class MultiscaleDiscriminator(object):
    def __init__(self, ndf=64, n_layers=3, norm_layer='instance',
                 use_sigmoid=False, num_D=2, getIntermFeat=True, name='discriminator'):
        self.name = name
        self.num_D = num_D
        self.n_layers = n_layers
        self.getIntermFeat = getIntermFeat
        if norm_layer=='instance':
            self.norm_layer = instance_norm 

        self.netD = []
        for i in range(num_D):
            self.netD.append(NLayerDiscriminator(ndf, n_layers, norm_layer, use_sigmoid, 
                                                 return_feat=getIntermFeat, name='discriminator_%d'%i))

        self.downsample = down_sample

    def forward(self, x, cond=None, reuse=False):
        num_D = self.num_D
        result = []
        input_downsampled = x
        for i in range(num_D):
            result.append(self.netD[i].forward(input_downsampled, cond, reuse=reuse))
            input_downsampled = self.downsample(input_downsampled)
        return result

# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(object):
    def __init__(self, ndf=64, n_layers=3, norm_layer='instance', use_sigmoid=False, return_feat=True, name='discriminator'):
        self.name = name
        self.n_layers = n_layers
        self.use_sigmoid = use_sigmoid
        self.ndf = ndf
        self.return_feat = return_feat
        if norm_layer=='instance':
            self.norm_layer = instance_norm 

    def forward(self, x, cond=None, reuse=False):
        
        with tf.variable_scope(self.name, reuse=reuse):
            if not cond is None:
                x = tf.concat((x, cond),axis=3)
            
            if self.return_feat: feats = []
            kw = 4
            padw = int(np.ceil((kw-1.0)/2))
            x = conv(x, self.ndf, kernel=kw, stride=2, pad=padw, pad_type='zero', scope='input_layer')
            x = lrelu(x, alpha=0.2)
            if self.return_feat: feats.append(x)

            nf = self.ndf
            for n in range(1, self.n_layers):
                nf = min(nf * 2, 512)
                x = conv(x, nf, kernel=kw, stride=2, pad=padw, pad_type='zero', scope='disc_layer_%d'%n)
                x = self.norm_layer(x, scope='norm_%d'%n)
                x = lrelu(x, alpha=0.2)
                if self.return_feat: feats.append(x)

            nf = min(nf * 2, 512)
            x = conv(x, nf, kernel=kw, stride=1, pad=padw, pad_type='zero', scope='disc_layer_%d'%(n+1))
            x = self.norm_layer(x, scope='norm_last')
            x = lrelu(x, alpha=0.2)
            if self.return_feat: feats.append(x)

            x = conv(x, 1, kernel=kw, stride=1, pad=padw, pad_type='zero', scope='disc_layer_out')

            if self.use_sigmoid:
                x = sigmoid(x)
            if self.return_feat: feats.append(x)    
        
            if self.return_feat:
                return feats
            else:
                return feats[-1]