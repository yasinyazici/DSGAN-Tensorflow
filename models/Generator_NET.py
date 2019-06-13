import tensorflow as tf
from lib.ops import * 

class GlobalTwoStreamGenerator(object):
    def __init__(self, output_nc=3, ngf=64, z_dim=16, n_downsampling=3, n_blocks=3, norm_layer='instance',
                 padding_type='reflect', use_skip=False, which_stream='ctx', use_output_gate=True,
                 feat_fusion='early_add', extra_embed=False, name='generator'):
        assert(n_blocks >= 0)
        assert( not (not ('label' in which_stream) and ('late' in feat_fusion)))
        assert( not (not ('ctx' in which_stream) and ('late' in feat_fusion)))
        if norm_layer=='instance':
            self.norm_layer = instance_norm 
        self.ngf = ngf
        self.z_dim=z_dim
        self.n_downsampling = n_downsampling
        self.padding_type=padding_type
        self.activation = relu
        self.output_nc = output_nc
        self.n_blocks = n_blocks
        self.use_skip = use_skip
        self.which_stream = which_stream
        self.use_output_gate=use_output_gate
        self.feat_fusion = feat_fusion
        feat_dim = self.ngf*2**n_downsampling
        self.feat_dim = feat_dim
        self.extra_embed = extra_embed
        self.name = name

    def forward(self, img, noise, mask, reuse=False):
        
        with tf.variable_scope(self.name, reuse=reuse):
            enc_feats = []
            x = img
            # input later
            x = conv(x, self.ngf, kernel=7, stride=1, pad=0, pad_type='reflect', scope='input')
            x = self.norm_layer(x, scope='instance_norm_input')
            x = self.activation(x)
        
            #down sampling layers
            for i in range(self.n_downsampling):
                mult = (2**i)
                x = conv(x, self.ngf * mult * 2, kernel=3, stride=2, pad=1, pad_type='zero', scope='down_sample_%d'%(i))
                x = self.norm_layer(x, scope='instance_norm_down_%d'%(i))
                x = self.activation(x)
                if self.use_skip and ((i < self.n_downsampling*3-1) and (i % 3 == 2)): # super-duper hard-coded
                    enc_feats.append(x)

            #noise layer
            noise = tf.expand_dims(tf.expand_dims(noise, 1), 1)
            h = x.get_shape().as_list()[1]
            noise = tf.concat([noise]*h,axis=1)
            noise = tf.concat([noise]*h,axis=2)
            x = tf.concat((x, noise),axis=3)
            x = conv(x, self.feat_dim, kernel=3, stride=1, pad=1, pad_type='zero', scope='noise_layer')
            x = self.norm_layer(x, scope='instance_norm_noise')
            x = self.activation(x)
        
            # embedding layers
            for i in range(self.n_blocks):
                x = ResnetBlock(x, self.feat_dim, pad_type=self.padding_type, norm_layer=self.norm_layer, 
                                use_dropout=False, scope='embedding_layer_%d'%(i))
            #x = self.activation(x)
            #x = self.norm_layer(x, scope='instance_norm_embedding')
            
            # up sample layers
            for i in range(self.n_downsampling):
                if (self.use_skip and len(enc_feats) > 0) and ((i > 0) and (i % 3 ==0)): # super-duper hard-coded
                    x = tf.concat((enc_feats[-int((i-3)/3)-1], x),axis=3)                    
                dim = int(self.ngf * (2**(self.n_downsampling - i)) / 2)
                x = deconv(x, dim, kernel=3, stride=2, padding='SAME', scope='up_sample%d'%(i))
                x = self.norm_layer(x, scope='instance_norm_up_%d'%(i))
                x = self.activation(x)
            # output layer
            x = conv(x, self.output_nc, kernel=7, stride=1, pad=3, pad_type='reflect', scope='output')
            x = tanh(x)
            output = x
        
            if self.use_output_gate:
                mask_output = tf.concat([mask]*self.output_nc, axis=-1)
                output = (1-mask_output)*img + mask_output*output

            return output