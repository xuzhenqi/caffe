import copy


class Atom:
    def __init__(self, title, value, isstring = False):
        self.title = title
        self.value = value
        self.isstring = isstring

    def display(self, level):
        string = level * "  " + self.title + ": "
        if self.isstring:
            string += "\""
        string += str(self.value)
        if self.isstring:
            string += "\""
        string += "\n"
        return string


class Compound:
    def __init__(self):
        None

    def initial(self, kwargs):
        for key in self.__titles__:
            # self.name = []
            # if 'name' in kwargs.keys(): self.name = kwargs['name']
            exec("self." + key + " = []");
            exec("if '" + key + "' in kwargs.keys(): self." + key +
                 " = copy.deepcopy(kwargs['" + key + "'])")

    def display(self, level=0):
        string = " {\n"
        level += 1
        for key in self.__titles__:
            # for v in self.name:
            #     string += v.display(level)
            exec("for v in self." + key + ":\n  if not isinstance(v, Atom):"
                                          "\n    string += level * '  ' + '"
                 + key + "'\n  string += v.display(level)")
        string += (level - 1) * "  " + "}\n"
        return string


class ParamSpec(Compound):
    def __init__(self, lr_mult, decay_mult, **kwargs):
        self.__titles__ = ['name', 'share_mode', 'lr_mult', 'decay_mult']
        self.initial(kwargs)
        self.lr_mult = [Atom('lr_mult', lr_mult)]
        self.decay_mult = [Atom('decay_mult', decay_mult)]

    def set_name(self, name):
        self.name = [Atom('name', name, True)]


class BlobProto(Compound):
    def __init__(self, *args, **kwargs):
        self.__titles__ = ['shape', 'data', 'diff', 'num', 'channels',
                           'height', 'width']
        self.initial(kwargs)


class BlobShape(Compound):
    def __init__(self, *args):
        self.__titles__ = ['dim']
        self.initial()
        for i in args:
            self.dim.append([Atom('dim', i)])


class NetStateRule(Compound):
    def __init__(self, **kwargs):
        self.__titles__ = ['phase', 'min_level', 'max_level', 'stage',
                           'not_stage']
        self.initial(kwargs)


class TransformationParameter(Compound):
    def __init__(self, **kwargs):
        self.__titles__ = ['scale', 'mirror', 'crop_size', 'mean_file',
                           'mean_value']
        self.initial(kwargs)


class LossParameter(Compound):
    def __init__(self, **kwargs):
        self.__titles__ = ['ignore_label', 'normalize']
        self.initial(kwargs)


class AccuracyParameter(Compound):
    def __init__(self, **kwargs):
        self.__titles__ = ['top_k', 'axis', 'ignore_label']
        self.initial(kwargs)


class ConcatParameter(Compound):
    def __init__(self, **kwargs):
        self.__titles__ = ['axis', 'concat_dim']
        self.initial(kwargs)


class ConvolutionParam(Compound):
    def __init__(self, num_output, pad, kernel_size, stride, **kwargs):
        self.__titles__ = ['num_output', 'bias_term', 'pad', 'pad_h', 'pad_w',
                           'kernel_size', 'kernel_h', 'kernel_w', 'group',
                           'stride', 'stride_h', 'stride_w', 'weight_filler',
                           'bias_filler', 'engine']
        self.initial(kwargs)
        self.num_output = [Atom('num_output', num_output)]
        self.pad = [Atom('pad', pad)]
        self.kernel_size = [Atom('kernel_size', kernel_size)]
        self.stride = [Atom('stride', stride)]


class FillerParameter(Compound):
    def __init__(self, type, *args, **kwargs):
        self.__titles__ = ['type', 'value', 'min', 'max', 'mean', 'std',
                           'sparse']
        self.initial(kwargs)
        self.type = [Atom('type', type, True)]
        if type == 'constant':
            self.value = [Atom('value', args[0])]
        elif type == 'xavier':
            None
        else:
            print type, 'filler not implented'
            exit(0)


class DataParameter(Compound):
    def __init__(self, **kwargs):
        self.__titles__ = ['source', 'batch_size', 'rand_skip', 'backend',
                           'scale', 'mean_file', 'crop_size', 'mirror',
                           'force_encodec_color']
        self.initial(kwargs)


class DropoutParameter(Compound):
    def __init__(self, dropout_ratio = 0.5):
        self.__titles__ = ['dropout_ratio']
        self.dropout_ratio = [Atom('dropout_ratio', dropout_ratio)]


class ImageDataParameter(Compound):
    def __init__(self, source, batch_size, root_folder, **kwargs):
        self.__titles__ = ['source', 'batch_size', 'rand_skip', 'shuffle',
                           'new_height', 'new_width', 'is_color', 'scale',
                           'mean_file', 'crop_size', 'mirror', 'root_folder']
        self.initial(kwargs)
        self.source = [Atom('source', source, True)]
        self.batch_size = [Atom('batch_size', batch_size)]
        self.root_folder = [Atom('root_folder', root_folder, True)]


class InnerProductParameter(Compound):
    def __init__(self, num_output, **kwargs):
        self.__titles__ = ['num_output', 'bias_term', 'weight_filler',
                           'bias_filler', 'axis']
        self.initial(kwargs)
        self.num_output = [Atom('num_output', num_output)]


class L2NParameter(Compound):
    def __init__(self, eps = 1e-5):
        self.__titles__ = ['eps']
        self.eps = [Atom('eps', eps)]


class LRNParameter(Compound):
    def __init__(self, local_size, alpha, beta, **kwargs):
        self.__titles__ = ['local_size', 'alpha', 'beta', 'norm_region', 'k']
        self.initial(kwargs)
        self.local_size = [Atom('local_size', local_size)]
        self.alpha = [Atom('alpha', alpha)]
        self.beta = [Atom('beta', beta)]


class LSTMParameter(Compound):
    def __init__(self, **kwargs):
        self.__titles__ = ['num_output', 'clipping_threshold', 'weight_filler',
                           'bias_filler', 'batch_size']
        self.initial(kwargs)


class PoolingParameter(Compound):
    def __init__(self, pool, kernel_size, stride, pad=0, **kwargs):
        self.__titles__ = ['pool', 'pad', 'pad_h', 'pad_w', 'kernel_size',
                           'kernel_h', 'kernel_w', 'stride', 'stride_h',
                           'kernel_w', 'engine', 'global_pooling']
        self.initial(kwargs)
        self.pool = [Atom('pool', pool)]
        self.kernel_size = [Atom('kernel_size', kernel_size)]
        self.stride = [Atom('stride', stride)]
        self.pad = [Atom('pad', pad)]


class ReLUParameter(Compound):
    def __init__(self, **kwargs):
        self.__titles__ = ['negative_slope', 'engine']
        self.initial(kwargs)


class TripletLossParameter(Compound):
    def __init__(self, margin=0.2):
        self.__titles__ = ['margin']
        self.margin = [Atom('margin', margin)]


class TripletImageDataParameter(Compound):
    def __init__(self, statics=''):
        self.__titles__ = ['statics']
        self.statics = [Atom('statics', statics, True)]


class LayerParameter(Compound):
    def __init__(self, type, **kwargs):
        self.__titles__ = ['name', 'type', 'bottom', 'top', 'phase',
                           'loss_weight', 'param', 'blobs', 'include',
                           'exclude', 'transform_param', 'loss_param',
                           'concat_param', 'convolution_param', 'data_param',
                           'dropout_param', 'image_data_param',
                           'inner_product_param', 'l2n_param', 'lrn_param',
                           'lstm_param', 'pooling_param', 'relu_param',
                           'triplet_loss_param', 'triplet_image_data_param']
        self.initial(kwargs)
        self.type = [Atom('type', type, True)]

    def set_name(self, name):
        self.name = [Atom('name', name, True)]

    def set_bottom(self, bottom=[]):
        self.bottom = []
        for i in bottom:
            self.bottom.append(Atom('bottom', i, True))

    def set_top(self, top=[]):
        self.top = []
        for i in top:
            self.top.append(Atom('top', i, True))


class NetParameter(Compound):
    def __init__(self, **kwargs):
        self.__titles__ = ['name', 'input', 'input_shape', 'input_dim',
                           'force_backward', 'state', 'debug_info', 'layer',
                           'layers']
        self.initial(kwargs)


class NetState(Compound):
    def __init__(self, **kwargs):
        self.__titles__ = ['phase', 'level', 'stage']
        self.initial(kwargs)


class Block():
    def __init__(self):
        None

    def display(self):
        return display(self.__layers__)


def display(layers):
    string = ''
    for v in layers:
        string += 'layer' + v.display()
    return string


class Fully_relu(Block):
    def __init__(self, name='', bottom='', top='', num_output=0):
        self.name = name
        self.bottom = bottom
        self.top = top
        self.num_output = num_output
        self.change()
        self.__layers__ = [self.inner, self.relu]

    def change(self):
        inner_product_param = InnerProductParameter(self.num_output,
                weight_filler = [filler_xavier_0_1],
                bias_filler = [filler_constant_0_2])
        self.inner = LayerParameter('InnerProduct', 
                param = [param_1_1, param_2_0],
                inner_product_param=[inner_product_param])
        self.inner.set_name(self.name + '/inner_product')
        self.inner.set_bottom([self.bottom])
        self.inner.set_top([self.top])
        self.relu = LayerParameter('ReLU')
        self.relu.set_name(self.name + '/relu')
        self.relu.set_bottom([self.top])
        self.relu.set_top([self.top])

    def set_name(self, name):
        self.name = name
        self.conv.set_name(self.name + '/conv')
        self.relu.set_name(self.name + '/relu')

    def set_bottom(self, bottom):
        self.bottom = bottom
        self.conv.set_bottom([self.bottom])

    def set_top(self, top):
        self.top = top
        self.conv.set_top([self.top])
        self.relu.set_bottom([self.top])
        self.relu.set_top([self.top])
    

class Conv_relu(Block):
    def __init__(self, name='', bottom='', top='', param=[]):
        self.name = name
        self.bottom = bottom
        self.top = top
        self.param = param
        self.change()
        self.__layers__ = [self.conv, self.relu]
    def change(self):
        conv_param = ConvolutionParam(self.param[0], self.param[1],
                                      self.param[2], self.param[3],
                                      weight_filler = [filler_xavier_0_1],
                                      bias_filler = [filler_constant_0_2])
        self.conv = LayerParameter('Convolution',
                        param = [param_1_1, param_2_0],
                        convolution_param =[conv_param])
        self.conv.set_name(self.name + '/conv')
        self.conv.set_bottom([self.bottom])
        self.conv.set_top([self.top])
        self.relu = LayerParameter('ReLU')
        self.relu.set_name(self.name + '/relu')
        self.relu.set_bottom([self.top])
        self.relu.set_top([self.top])

    def set_name(self, name):
        self.name = name
        self.conv.set_name(self.name + '/conv')
        self.relu.set_name(self.name + '/relu')

    def set_bottom(self, bottom):
        self.bottom = bottom
        self.conv.set_bottom([self.bottom])

    def set_top(self, top):
        self.top = top
        self.conv.set_top([self.top])
        self.relu.set_bottom([self.top])
        self.relu.set_top([self.top])


class Inception(Block):
    def __init__(self, name='', bottom='', top='', param=[]):
        self.c1x1_param = param[0]
        self.c3x3_r_param = param[1]
        self.c3x3_param = param[2]
        self.c5x5_r_param = param[3]
        self.c5x5_param = param[4]
        self.pool_param = param[5]
        self.pool_pro_param = param[6]
        self.name = name
        self.bottom = bottom
        self.top = top
        self.__layers__ = []
        self.change()

    def change(self):
        self.concat_bottom = []
        if self.c1x1_param:
            self.c1x1_relu = Conv_relu(self.name + '/c1x1',
                                       self.bottom,
                                       self.name + '/c1x1',
                                       self.c1x1_param)
            self.__layers__ += self.c1x1_relu.__layers__
            self.concat_bottom.append(self.name + '/c1x1')
        else:
            print "Info: 1x1 conv not set"
        bool_temp = bool(self.c3x3_r_param) + bool(self.c3x3_param)
        if  bool_temp == 1:
            print "Error: c3x3_r_param and c3x3_param \
                    should be set both or not set both"
            exit(0)
        elif bool_temp == 0:
            print "Info: 3x3 conv not set"
        else:
            self.c3x3_relu_r = Conv_relu(self.name + '/c3x3_r',
                                       self.bottom,
                                       self.name + '/c3x3_r',
                                       self.c3x3_r_param)
            self.__layers__ += self.c3x3_relu_r.__layers__
            self.c3x3_relu = Conv_relu(self.name + '/c3x3',
                                       self.name + '/c3x3_r',
                                       self.name + '/c3x3',
                                       self.c3x3_param)
            self.__layers__ += self.c3x3_relu.__layers__
            self.concat_bottom.append(self.name + '/c3x3')
        bool_temp = bool(self.c5x5_r_param) + bool(self.c5x5_param)
        if  bool_temp == 1:
            print "Error: c5x5_r_param and c5x5_param \
                    should be set both or not set both"
            exit(0)
        elif bool_temp == 0:
            print "Info: 5x5 conv not set"
        else:
            self.c5x5_relu_r = Conv_relu(self.name + '/c5x5_r',
                                         self.bottom,
                                         self.name + '/c5x5_r',
                                         self.c5x5_r_param)
            self.__layers__ += self.c5x5_relu_r.__layers__
            self.c5x5_relu = Conv_relu(self.name + '/c5x5',
                                       self.name + '/c5x5_r',
                                       self.name + '/c5x5',
                                       self.c5x5_param)
            self.__layers__ += self.c5x5_relu.__layers__
            self.concat_bottom.append(self.name + '/c5x5')
        if self.pool_param:
            pool_param = PoolingParameter(self.pool_param[0], self.pool_param[1],
                                          self.pool_param[2], self.pool_param[3])
            self.pool_layer = LayerParameter('Pooling',
                                             pooling_param = [pool_param])
            self.pool_layer.set_name(self.name + '/pool')
            self.pool_layer.set_bottom([self.bottom])
            self.pool_layer.set_top([self.name + '/pool'])
            self.__layers__.append(self.pool_layer)
            self.concat_bottom.append(self.name + '/pool')
            if self.pool_pro_param:
                self.pool_pro = Conv_relu(self.name + '/pool_proj',
                                          self.name + '/pool',
                                          self.name + '/pool_proj',
                                          self.pool_pro_param)
                self.__layers__ += self.pool_pro.__layers__
                self.concat_bottom[-1] = self.name + '/pool_proj'
        elif self.pool_pro_param:
            print "pool_pro exists only if pool exists!"
            exit(0)
        if len(self.concat_bottom) <= 1:
            print "Error: concat_bottom should be more than 1"
            exit(0)
        self.concat_layer = LayerParameter('Concat')
        self.concat_layer.set_name(self.name + '/concat')
        self.concat_layer.set_bottom(self.concat_bottom)
        self.concat_layer.set_top([self.top])
        self.__layers__.append(self.concat_layer)

    def set_name(self, name):
        self.name = name
        for i in self.__layers__:
            i.set_name(name)

    def set_bottom(self, bottom):
        self.bottom = bottom
        if (self.name + '/c1x1') in self.concat_bottom:
            self.c1x1_relu.set_bottom([bottom])
        if (self.name + '/c3x3') in self.concat_bottom:
            self.c3x3_relu_r.set_bottom([bottom])
        if (self.name + '/c5x5') in self.concat_bottom:
            self.c5x5_relu_r.set_bottom([bottom])
        if (self.name + '/pool_proj') in self.concat_bottom:
            self.pool_layer.set_bottom([bottom])
        if (self.name + '/pool') in self.concat_bottom:
            self.pool_layer.set_bottom([bottom])

    def set_top(self, top):
        self.top = top
        self.concat_layer.set_top([top])


def copy_layers(layers=[], suffix1='', suffix2='_2', share_param=False):
    if layers == []:
        return []
    cp_layers = copy.deepcopy(layers)
    if share_param:
        for i in range(len(layers)):
            if 'param' not in layers[i].__titles__:
                continue
            param = layers[i].param
            cp_param = cp_layers[i].param
            for j in range(len(param)):
                name = 'p' + str(j) + layers[i].name[0].value
                if param[j].name == []:
                    param[j].set_name(name)
                    cp_param[j].set_name(name)
    if suffix1 != '':
        for i in layers:
            i.set_name(i.name[0].value + suffix1)
            bottom = []
            for j in i.bottom:
                bottom.append(j.value + suffix1)
            i.set_bottom(bottom)
            top = []
            for j in i.top:
                top.append(j.value + suffix1)
            i.set_top(top)

    if suffix2 != '':
        for i in cp_layers:
            i.set_name(i.name[0].value + suffix2)
            bottom = []
            for j in i.bottom:
                bottom.append(j.value + suffix2)
            i.set_bottom(bottom)
            top = []
            for j in i.top:
                top.append(j.value + suffix2)
            i.set_top(top)
    return (layers, cp_layers)

dataset = '/mnt/dataset2/CASIAWebFace/'
image_data_param = ImageDataParameter(dataset + 'filelist_crop.txt',
                                      60,
                                      dataset + 'casia_crop/')
triplet_image_data_param = TripletImageDataParameter(
    dataset + 'identities.txt')
param_1_1 = ParamSpec(1, 1)
param_2_0 = ParamSpec(2, 0)
filler_constant_0_2 = FillerParameter('constant', 0.2)
filler_xavier_0_1 = FillerParameter('xavier', 0.1)
convolution_param_64_1_3_1 = ConvolutionParam(64, 1, 3, 1)
convolution_param_64_0_1_1 = ConvolutionParam(64, 0, 1, 1)
pooling_param_max_3_2 = PoolingParameter('MAX', 3, 2)
lrn_param_5_0_0001_0_75 = LRNParameter(5, 0.0001, 0.75)

if __name__ == '__main__':
    # image_data_layer
    triplet_image_data_layer = LayerParameter('TripletImageData',
                    image_data_param = [image_data_param],
                    triplet_image_data_param = [triplet_image_data_param])
    triplet_image_data_layer.set_name('triplet_data_layer')
    triplet_image_data_layer.set_top(['data_1', 'data_2', 'data_3'])
    
    googlenet_layers = []
    #print triplet_image_data_layer.display(0)
    conv1 = Conv_relu('conv1', 'data', 'conv1', param=[64, 1, 3, 1])
    googlenet_layers += conv1.__layers__
    #print conv1.display()
    max_pool_layer = LayerParameter('Pooling',
            pooling_param = [pooling_param_max_3_2])
    max_pool_layer.set_name('max_pool')
    max_pool_layer.set_bottom(['conv1'])
    max_pool_layer.set_top(['max_pool'])
    googlenet_layers.append(max_pool_layer)
    #print max_pool_layer.display(0)
    lrn_layer = LayerParameter('LRN', lrn_param = [lrn_param_5_0_0001_0_75])
    lrn_layer.set_name('lrn')
    lrn_layer.set_bottom(['max_pool'])
    lrn_layer.set_top(['lrn'])
    googlenet_layers.append(lrn_layer)
    #print lrn_layer.display(0)
    inception2_3x3_r = Conv_relu('inception2/c3x3_r', 'lrn', 
            'inception2/c3x3_r', param=[64, 0, 1, 1])
    inception2_3x3 = Conv_relu('inception2/c3x3', 'inception2/c3x3_r',
            'inception2', param=[192, 1, 3, 1])
    googlenet_layers += inception2_3x3_r.__layers__ + inception2_3x3.__layers__
    #print inception2_3x3_r.display()
    #print inception2_3x3.display()
    lrn2_layer = LayerParameter('LRN', lrn_param = [lrn_param_5_0_0001_0_75])
    lrn2_layer.set_name('lrn2')
    lrn2_layer.set_bottom(['inception2'])
    lrn2_layer.set_top(['lrn2'])
    googlenet_layers.append(lrn2_layer)
    #print lrn2_layer.display(0)
    max_pool2_layer = LayerParameter('Pooling',
                                   pooling_param = [pooling_param_max_3_2])
    max_pool2_layer.set_name('max_pool2')
    max_pool2_layer.set_bottom(['lrn2'])
    max_pool2_layer.set_top(['max_pool2'])
    googlenet_layers.append(max_pool2_layer)
    #print max_pool2_layer.display(0)
    inception_3a = Inception('inception_3a', 'max_pool2', 'inception_3a',
                             [[64, 0, 1, 1], [96, 0, 1, 1], [128, 1, 3, 1],
                             [16, 0, 1, 1], [32, 2, 5, 1], ['MAX', 3, 1, 1],
                             [32, 0, 1, 1]])
    inception_3b = Inception('inception_3b', 'inception_3a', 'inception_3b',
                             [[64, 0, 1, 1], [96, 0, 1, 1], [128, 1, 3, 1],
                             [16, 0, 1, 1], [32, 2, 5, 1], ['L2NORM', 3, 1, 1],
                             [64, 0, 1, 1]])
    inception_3c = Inception('inception_3c', 'inception_3b', 'inception_3c',
                             [None, [128, 0, 1, 1], [256, 1, 3, 2],
                             [32, 0, 1, 1], [64, 2, 5, 2], ['MAX', 3, 2, 0],
                             None])
    googlenet_layers += inception_3a.__layers__ + inception_3b.__layers__ +\
        inception_3c.__layers__ 
    #print inception_3c.display()
    inception_4a = Inception('inception_4a', 'inception_3c', 'inception_4a',
                             [[256, 0, 1, 1], [96, 0, 1, 1], [192, 1, 3, 1],
                             [32, 0, 1, 1], [64, 2, 5, 1], ['L2NORM', 3, 1, 1],
                             [128, 0, 1, 1]])
    inception_4b = Inception('inception_4b', 'inception_4a', 'inception_4b',
                             [[224, 0, 1, 1], [112, 0, 1, 1], [224, 1, 3, 1],
                             [32, 0, 1, 1], [64, 2, 5, 1], ['L2NORM', 3, 1, 1],
                             [128, 0, 1, 1]])
    inception_4c = Inception('inception_4c', 'inception_4b', 'inception_4c',
                             [[192, 0, 1, 1], [128, 0, 1, 1], [256, 1, 3, 1],
                             [32, 0, 1, 1], [64, 2, 5, 1], ['L2NORM', 3, 1, 1],
                             [128, 0, 1, 1]])
    inception_4d = Inception('inception_4d', 'inception_4c', 'inception_4d',
                             [[160, 0, 1, 1], [144, 0, 1, 1], [288, 1, 3, 1],
                             [32, 0, 1, 1], [64, 2, 5, 1], ['L2NORM', 3, 1, 1],
                             [128, 0, 1, 1]])
    inception_4e = Inception('inception_4e', 'inception_4d', 'inception_4e',
                             [None, [160, 0, 1, 1], [256, 1, 3, 2],
                             [64, 0, 1, 1], [128, 2, 5, 2], ['MAX', 3, 2, 0],
                             None])

    googlenet_layers += inception_4a.__layers__ + inception_4b.__layers__ +\
        inception_4c.__layers__ + inception_4d.__layers__ + \
        inception_4e.__layers__ 
    inception_5a = Inception('inception_5a', 'inception_4e', 'inception_5a',
                             [[384, 0, 1, 1], [192, 0, 1, 1], [384, 1, 3, 1],
                             [48, 0, 1, 1], [182, 2, 5, 1], ['L2NORM', 3, 1, 1],
                             [128, 0, 1, 1]])
    inception_5b = Inception('inception_5b', 'inception_5a', 'inception_5b',
                             [[384, 0, 1, 1], [192, 0, 1, 1], [384, 1, 3, 1],
                             [48, 0, 1, 1], [182, 2, 5, 1], ['MAX', 3, 1, 1],
                             [128, 0, 1, 1]])
    googlenet_layers += inception_5a.__layers__ + inception_5b.__layers__
    avg_pool_param = PoolingParameter('AVE', 6, 1)
    #print avg_pool_param.display(0)
    avg_pool_layer = LayerParameter('Pooling', 
            pooling_param = [avg_pool_param])
    avg_pool_layer.set_name('avg_pool')
    avg_pool_layer.set_bottom(['inception_5b'])
    avg_pool_layer.set_top(['avg_pool'])
    googlenet_layers.append(avg_pool_layer)
    #print avg_pool_layer.display(0)
    fully_conn = Fully_relu('fully_conn', 'avg_pool', 'fully_conn', 128)
    googlenet_layers += fully_conn.__layers__
    #print fully_conn.display()
   
    (googlenet_layers, googlenet_1) = copy_layers(googlenet_layers, '', '_1', True)
    (googlenet_layers, googlenet_2) = copy_layers(googlenet_layers, '', '_2', True)
    (googlenet_layers, googlenet_3) = copy_layers(googlenet_layers, '', '_3', True)
    #print inception_3a.display()
    #print display(Inception_3a_2)
    triplet_loss_param = TripletLossParameter(0.4)
    triplet_loss_layer = LayerParameter('TripletLoss',
            triplet_loss_param = [triplet_loss_param])
    triplet_loss_layer.set_name('triplet_loss')
    triplet_loss_layer.set_bottom(['fully_conn_1', 'fully_conn_2', 'fully_conn_3'])
    triplet_loss_layer.set_top(['loss'])
    #print triplet_loss_layer.display(0)
    print display([triplet_image_data_layer] + googlenet_1 + googlenet_2 + googlenet_3\
            + [triplet_loss_layer])

