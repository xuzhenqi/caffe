import copy
import traceback

class Atom:
    def __init__(self, title, value):
        self.title = title
        self.value = value
        self.isstring = isinstance(value, str)

    def display(self, level):
        string = level * "  " + self.title + ": "
        if self.isstring:
            string += "\""
        if isinstance(self.value, bool):
            if self.value:
                string += "true"
            else:
                string += 'false'
        else:
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
            # if 'name' in kwargs.keys(): self.name = Atom('name', kwargs['name'])
            exec ("self." + key + " = []");
            if key in kwargs.keys():
                self.set(key, kwargs[key])

    def set(self, key, value):
        if key not in self.__titles__:
            print "[Warning] %s is not a valid field" % key
        else:
            exec ('self.%s = []' % key)
            exec ("if isinstance(value, list):\n" +
                  "  for v in value:\n" +
                  "    if isinstance(v, Compound):\n" +
                  "      self.%s += [v]\n" % key +
                  "    else:\n" +
                  "      self.%s += [Atom('%s', v)]\n" % (key, key) +
                  "else:\n" +
                  "  if isinstance(value, Compound):\n" +
                  "    self.%s = value\n" % key +
                  "  else:\n" +
                  "    self.%s = Atom('%s', value)\n" % (key, key))


    def display(self, level=0):
        string = " {\n"
        level += 1
        for key in self.__titles__:
            # for v in self.name:
            # string += v.display(level)
            exec ("if not isinstance(self.%s, list):\n" % key +
                  "  self.%s = [self.%s]\n" % (key, key) +
                  "for v in self.%s:\n" % key +
                  "  if not isinstance(v, Atom):\n" +
                  "    string += level * '  ' + '%s'\n" % key +
                  "  string += v.display(level)\n"
                  )
        string += (level - 1) * "  " + "}\n"
        return string


class ParamSpec(Compound):
    def __init__(self, lr_mult, decay_mult, **kwargs):
        self.__titles__ = ['name', 'share_mode', 'lr_mult', 'decay_mult']
        self.initial(kwargs)
        self.lr_mult = [Atom('lr_mult', lr_mult)]
        self.decay_mult = [Atom('decay_mult', decay_mult)]

    def set_name(self, name):
        self.name = [Atom('name', name)]


class BlobProto(Compound):
    def __init__(self, *args, **kwargs):
        self.__titles__ = ['shape', 'data', 'diff', 'num', 'channels',
                           'height', 'width']
        self.initial(kwargs)


class BlobShape(Compound):
    def __init__(self, *args):
        self.__titles__ = ['dim']
        self.initial({})
        for i in args:
            self.dim.append(Atom('dim', i))


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
    def __init__(self, type, **kwargs):
        self.__titles__ = ['type', 'value', 'min', 'max', 'mean', 'std',
                           'sparse']
        self.initial(kwargs)
        self.type = [Atom('type', type)]


class DataParameter(Compound):
    def __init__(self, **kwargs):
        self.__titles__ = ['source', 'batch_size', 'rand_skip', 'backend',
                           'scale', 'mean_file', 'crop_size', 'mirror',
                           'force_encodec_color']
        self.initial(kwargs)


class DropoutParameter(Compound):
    def __init__(self, dropout_ratio=0.5):
        self.__titles__ = ['dropout_ratio']
        self.dropout_ratio = [Atom('dropout_ratio', dropout_ratio)]


class ImageDataParameter(Compound):
    def __init__(self, source, batch_size, root_folder, **kwargs):
        self.__titles__ = ['source', 'batch_size', 'rand_skip', 'shuffle',
                           'new_height', 'new_width', 'is_color', 'scale',
                           'mean_file', 'crop_size', 'mirror', 'root_folder']
        self.initial(kwargs)
        self.source = [Atom('source', source)]
        self.batch_size = [Atom('batch_size', batch_size)]
        self.root_folder = [Atom('root_folder', root_folder)]


class InnerProductParameter(Compound):
    def __init__(self, num_output, **kwargs):
        self.__titles__ = ['num_output', 'bias_term', 'weight_filler',
                           'bias_filler', 'axis']
        self.initial(kwargs)
        self.num_output = [Atom('num_output', num_output)]


class L2NParameter(Compound):
    def __init__(self, eps=1e-5):
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
        self.statics = [Atom('statics', statics)]


class EltwiseParameter(Compound):
    def __init__(self, operation, **kwargs):
        self.__titles__ = ['operation', 'coeff', 'stable_prod_grad']
        self.initial(kwargs)
        self.operation = [Atom('operation', operation)]

class SumRNNParameter(Compound):
    def __init__(self, *args):
        self.__titles__ = ['coeff']
        self.initial({})
        for i in args:
            self.coeff.append(Atom('coeff', i))

class ImageDataRNNParameter(Compound):
    def __init__(self, fps):
        self.__titles__ = ['fps']
        self.fps = [Atom('fps', fps)]


class LayerParameter(Compound):
    def __init__(self, type, **kwargs):
        self.__titles__ = ['name', 'type', 'bottom', 'top', 'phase',
                           'loss_weight', 'param', 'blobs', 'propagate_down',
                           'include',
                           'exclude', 'bottom_from', 'bottom_shape',
                           'transform_param',
                           'loss_param',
                           'concat_param', 'convolution_param', 'data_param',
                           'eltwise_param', 'sum_rnn_param', 'accuracy_param',
                           'dropout_param', 'image_data_param',
                           'image_data_rnn_param',
                           'inner_product_param', 'l2n_param', 'lrn_param',
                           'lstm_param', 'pooling_param', 'relu_param',
                           'triplet_loss_param', 'triplet_image_data_param']
        self.initial(kwargs)
        self.type = [Atom('type', type)]

    def set_name(self, name):
        self.name = [Atom('name', name)]

    def set_bottom(self, bottom=[]):
        self.bottom = []
        for i in bottom:
            self.bottom.append(Atom('bottom', i))

    def set_top(self, top=[]):
        self.top = []
        for i in top:
            self.top.append(Atom('top', i))


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


class Enum:
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return self.value


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


param_1_1 = ParamSpec(1, 1)
param_2_0 = ParamSpec(2, 0)


class Fully_relu(Block):
    def __init__(self, name, bottom, top, inner_param, param=[param_1_1, param_2_0]):
        self.name = name
        self.bottom = bottom
        self.top = top
        self.inner_product_param = inner_param
        self.param = param
        self.change()
        self.__layers__ = [self.inner, self.relu]

    def change(self):
        self.inner = LayerParameter('InnerProduct',
                                    param=self.param,
                                    inner_product_param=[self.inner_product_param])
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
    def __init__(self, name, bottom, top, conv_param,
                 param=[param_1_1, param_2_0]):
        self.name = name
        self.bottom = bottom
        self.top = top
        self.conv_param = conv_param
        self.param = param
        self.change()
        self.__layers__ = [self.conv, self.relu]

    def change(self):
        self.conv = LayerParameter('Convolution',
                                   param=self.param,
                                   convolution_param=self.conv_param)
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


class ConvRNN(Block):
    def __init__(self, name, bottom, top, conv_param, eltwise_param, param):
        self.name = name
        self.bottom = bottom
        self.top = top
        self.conv_param = conv_param
        self.eltwise_param = eltwise_param
        self.param = param
        self.change()

    def change(self):
        # check parameters
        if not (isinstance(self.bottom, list) and len(self.bottom) == 2):
            print "[Error]: ConvRNN block receive two bottom blobs."
            exit(0)
        if not (isinstance(self.top, list) and len(self.top) == 2):
            print "[Error]: ConvRNN block receive two top blobs."
            exit(0)
        if not (isinstance(self.conv_param, list) and len(self.conv_param) == 2):
            print "[Error]: ConvRNN block receive two conv_params."
            exit(0)
        if not isinstance(self.param[0], list):
            self.param = [self.param, self.param]
        # Conv layer
        self.conv = LayerParameter('Convolution',
                                   name=self.name + '/conv',
                                   bottom=self.bottom,
                                   top=[self.top[0], self.name + '/conv2'],
                                   convolution_param=self.conv_param[0],
                                   param=self.param[0])
        # Relu1
        self.relu1 = LayerParameter('ReLU',
                                    name=self.name + '/relu1',
                                    bottom=self.top[0],
                                    top=self.top[0])
        # ConvRNN
        self.conv_rnn = LayerParameter('Convolution',
                                       name=self.name + '/convrnn',
                                       bottom=self.top[0],
                                       top=self.name + '/convrnn',
                                       convolution_param=self.conv_param[1],
                                       param=self.param[1])
        # Eltwise SUM
        self.sum = LayerParameter('Eltwise',
                                  name=self.name + '/sum',
                                  bottom=[self.name + '/conv2', self.name + '/convrnn'],
                                  top=self.top[1],
                                  eltwise_param=self.eltwise_param)
        # relu2
        self.relu2 = LayerParameter('ReLU',
                                    name=self.name + '/relu2',
                                    bottom=self.top[1],
                                    top=self.top[1])
        self.__layers__ = [self.conv, self.relu1, self.conv_rnn, self.sum, self.relu2]

class ConvRNN_v2(Block):
    def __init__(self, name, bottom, top, conv_param, sum_rnn_param, param,
                 rnn_shape):
        self.name = name
        self.bottom = bottom
        self.top = top
        self.conv_param = conv_param
        self.sum_rnn_param = sum_rnn_param
        self.param = param
        self.rnn_shape = rnn_shape
        self.change()

    def change(self):
        # check parameters
        if not (isinstance(self.conv_param, list) and len(self.conv_param) == 2):
            print "[Error]: ConvRNN_v2 block receive two conv_params."
            exit(0)
        if not (isinstance(self.param[0], list) and len(self.param[0]) == 2
                and not isinstance(self.param[1], list)):
            print "[Error]: ConvRNN_v2 block receive two params, one has two" \
                  " items and the other has one!"
            exit(0)
        # Conv layer
        self.conv = LayerParameter('Convolution',
                                   name=self.name + '/conv',
                                   bottom=self.bottom,
                                   top=self.name + '/conv',
                                   convolution_param=self.conv_param[0],
                                   param=self.param[0])
        # ConvRNN
        self.conv_rnn = LayerParameter('Convolution',
                                       name=self.name + '/convrnn',
                                       bottom=self.name + '_rnn',
                                       top=self.name + '/convrnn',
                                       convolution_param=self.conv_param[1],
                                       param=self.param[1],
                                       bottom_from=self.top,
                                       bottom_shape=self.rnn_shape,
                                       propagate_down=Enum('false'))

        # Eltwise SUM
        self.sum = LayerParameter('SumRNN',
            name=self.name + '/sum',
            bottom=[self.name + '/conv', self.name + '/convrnn',
                    'begin_marker'],
            top=self.top,
            sum_rnn_param=self.sum_rnn_param)
        # relu
        self.relu = LayerParameter('ReLU',
                                    name=self.name + '/relu',
                                    bottom=self.top,
                                    top=self.top)
        self.__layers__ = [self.conv, self.conv_rnn, self.sum, self.relu]

class IPRNN(Block):
    def __init__(self, name, bottom, top, ip_param, eltwise_param, param):
        self.name = name
        self.bottom = bottom
        self.top = top
        self.ip_param = ip_param
        self.eltwise_param = eltwise_param
        self.param = param
        self.change()

    def change(self):
        # check parameters
        if not (isinstance(self.bottom, list) and len(self.bottom) == 2):
            print "[Error]: IPRNN block receive two bottom blobs."
            exit(0)
        if not (isinstance(self.top, list) and len(self.top) == 2):
            print "[Error]: IPRNN block receive two top blobs."
            exit(0)
        if not (isinstance(self.ip_param, list) and len(self.ip_param) == 2):
            print "[Error]: IPRNN block receive two ip_params."
            exit(0)
        if not isinstance(self.param[0], list):
            self.param = [self.param, self.param]
        # IP layer
        self.ip = LayerParameter('InnerProduct',
                                 name=self.name + '/ip',
                                 bottom=self.bottom,
                                 top=[self.top[0], self.name + '/ip2'],
                                 inner_product_param=self.ip_param[0],
                                 param=self.param[0])
        # Relu1
        self.relu1 = LayerParameter('ReLU',
                                    name=self.name + '/relu1',
                                    bottom=self.top[0],
                                    top=self.top[0])
        # ConvRNN
        self.ip_rnn = LayerParameter('InnerProduct',
                                     name=self.name + '/iprnn',
                                     bottom=self.top[0],
                                     top=self.name + '/iprnn',
                                     inner_product_param=self.ip_param[1],
                                     param=self.param[1])
        # Eltwise SUM
        self.sum = LayerParameter('Eltwise',
                                  name=self.name + '/sum',
                                  bottom=[self.name + '/ip2', self.name + '/iprnn'],
                                  top=self.top[1],
                                  eltwise_param=self.eltwise_param)
        # relu2
        self.relu2 = LayerParameter('ReLU',
                                    name=self.name + '/relu2',
                                    bottom=self.top[1],
                                    top=self.top[1])
        self.__layers__ = [self.ip, self.relu1, self.ip_rnn, self.sum, self.relu2]

class IPRNN_v2(Block):
    def __init__(self, name, bottom, top, ip_param, sum_rnn_param, param,
                 rnn_shape):
        self.name = name
        self.bottom = bottom
        self.top = top
        self.ip_param = ip_param
        self.sum_rnn_param = sum_rnn_param
        self.param = param
        self.rnn_shape = rnn_shape
        self.change()

    def change(self):
        # check parameters
        if not (isinstance(self.ip_param, list) and len(self.ip_param) == 2):
            print "[Error]: IPRNN block receive two ip_params."
            exit(0)
        if not (isinstance(self.param[0], list) and len(self.param[0]) == 2
                and not isinstance(self.param[1], list)):
            print "[Error]: ConvRNN_v2 block receive two params, one has two" \
                  " items and the other has one!"
            exit(0)
        # IP layer
        self.ip = LayerParameter('InnerProduct',
                                 name=self.name + '/ip',
                                 bottom=self.bottom,
                                 top=self.name + '/ip',
                                 inner_product_param=self.ip_param[0],
                                 param=self.param[0])
        # IPRNN
        self.ip_rnn = LayerParameter('InnerProduct',
                                     name=self.name + '/iprnn',
                                     bottom=self.name + '_rnn',
                                     top=self.name + '/iprnn',
                                     inner_product_param=self.ip_param[1],
                                     param=self.param[1],
                                     bottom_from=self.top,
                                     bottom_shape=self.rnn_shape,
                                     propagate_down=False)
        # Eltwise SUM
        self.sum = LayerParameter('SumRNN',
               name=self.name + '/sum',
               bottom=[self.name + '/ip', self.name + '/iprnn', 'begin_marker'],
               top=self.top,
               sum_rnn_param=self.sum_rnn_param)
        # relu
        self.relu = LayerParameter('ReLU',
                                    name=self.name + '/relu',
                                    bottom=self.top,
                                    top=self.top)
        self.__layers__ = [self.ip, self.ip_rnn, self.sum, self.relu]

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
        if bool_temp == 1:
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
        if bool_temp == 1:
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
            self.pool_layer = LayerParameter('Pooling',
                                             pooling_param=self.pool_param)
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
            layer_name = i.name.split('/')
            for i in layer_name[1:]:
                name += '/' + i
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


class InceptionRNN(Block):
    def __init__(self, name, bottom, top, inception_param, eltwise_param,
                 param):
        self.name = name
        self.bottom = bottom
        self.top = top
        self.inception_param = inception_param
        self.eltwise_param = eltwise_param
        self.param = param
        self.__layers__ = []
        self.change()

    def change(self):
        # check param
        if not (isinstance(self.bottom, list) and len(self.bottom) == 2):
            print "[Error]: InceptionRNN block receive two bottom blobs."
            exit(0)
        if not (isinstance(self.top, list) and len(self.top) == 2):
            print "[Error]: InceptionRNN block receive two top blobs."
            exit(0)
        if not (isinstance(self.inception_param, list)
                and len(self.inception_param) == 7):
            print "[Error]: InceptionRNN block receive two conv_params."
            exit(0)
        if not isinstance(self.param[0], list):
            self.param = [self.param, self.param]
        self.__layers__ = []

        self.concat_bottom = []
        self.concat_rnn_bottom = []
        if self.inception_param[0]:
            self.c1x1_rnn = ConvRNN(self.name+'/c1x1_rnn', self.bottom,
                [self.name+'/c1x1_rnn_1', self.name+'/c1x1_rnn_2'],
                [self.inception_param[0], self.inception_param[0]],
                self.eltwise_param, self.param)
            self.__layers__ += self.c1x1_rnn.__layers__
            self.concat_bottom.append(self.name + '/c1x1_rnn_1')
            self.concat_rnn_bottom.append(self.name + '/c1x1_rnn_2')
        else:
            print "Info: 1x1 conv not set"
        bool_temp = bool(self.inception_param[1]) + bool(self.inception_param[2])
        if bool_temp == 1:
            print "Error: c3x3_r_param and c3x3_param \
                    should be set both or not set both"
            exit(0)
        elif bool_temp == 0:
            print "Info: 3x3 conv not set"
        else:
            self.c3x3_rnn_r = ConvRNN(self.name + '/c3x3_rnn_r', self.bottom,
                [self.name + '/c3x3_rnn_r_1', self.name + '/c3x3_rnn_r_2'],
                [self.inception_param[1], self.inception_param[1]],
                self.eltwise_param, self.param)
            self.__layers__ += self.c3x3_rnn_r.__layers__
            self.c3x3_rnn = ConvRNN(self.name + '/c3x3_rnn',
                [self.name + '/c3x3_rnn_r_1', self.name + '/c3x3_rnn_r_2'],
                [self.name + '/c3x3_rnn_1', self.name + '/c3x3_rnn_2'],
                [self.inception_param[2], self.inception_param[2]],
                self.eltwise_param, self.param)
            self.__layers__ += self.c3x3_rnn.__layers__
            self.concat_bottom.append(self.name + '/c3x3_rnn_1')
            self.concat_rnn_bottom.append(self.name + '/c3x3_rnn_2')
        bool_temp = bool(self.inception_param[3]) + bool(self.inception_param[4])
        if bool_temp == 1:
            print "Error: c5x5_r_param and c5x5_param \
                    should be set both or not set both"
            exit(0)
        elif bool_temp == 0:
            print "Info: 5x5 conv not set"
        else:
            self.c5x5_rnn_r = ConvRNN(self.name + '/c5x5_rnn_r', self.bottom,
                [self.name + '/c5x5_rnn_r_1', self.name + '/c5x5_rnn_r_2'],
                [self.inception_param[3], self.inception_param[3]],
                self.eltwise_param, self.param)
            self.__layers__ += self.c5x5_rnn_r.__layers__
            self.c5x5_rnn = ConvRNN(self.name + '/c5x5_rnn',
                [self.name + '/c5x5_rnn_r_1', self.name + '/c5x5_rnn_r_2'],
                [self.name + '/c5x5_rnn_1', self.name + '/c5x5_rnn_2'],
                [self.inception_param[4], self.inception_param[4]],
                self.eltwise_param, self.param)
            self.__layers__ += self.c5x5_rnn.__layers__
            self.concat_bottom.append(self.name + '/c5x5_rnn_1')
            self.concat_rnn_bottom.append(self.name + '/c5x5_rnn_2')
        if self.inception_param[5]:
            self.pool = LayerParameter('Pooling',
                name=self.name + '/pool_1',
                bottom=self.bottom[0],
                top=self.name + '/pool_1',
                pooling_param=self.inception_param[5])
            self.pool_rnn = LayerParameter('Pooling',
                name=self.name + '/pool_2',
                bottom=self.bottom[1],
                top=self.name + '/pool_2',
                pooling_param=self.inception_param[5])
            self.__layers__.append(self.pool)
            self.__layers__.append(self.pool_rnn)
            self.concat_bottom.append(self.name + '/pool_1')
            self.concat_rnn_bottom.append(self.name + '/pool_2')
            if self.inception_param[6]:
                self.pool_pro_rnn = ConvRNN(self.name + '/pool_pro_rnn',
                    [self.name + '/pool_1', self.name + '/pool_2'],
                    [self.name + '/pool_pro_1', self.name + '/pool_pro_2'],
                    [self.inception_param[6], self.inception_param[6]],
                    self.eltwise_param, self.param)
                self.__layers__ += self.pool_pro_rnn.__layers__
                self.concat_bottom[-1] = self.name + '/pool_pro_1'
                self.concat_rnn_bottom[-1] = self.name + '/pool_pro_2'
        elif self.inception_param[6]:
            print "pool_pro exists only if pool exists!"
            exit(0)
        if len(self.concat_bottom) <= 1 or len(self.concat_rnn_bottom) <=1:
            print "Error: concat_bottom should be more than 1"
            exit(0)
        self.concat = LayerParameter('Concat',
            name=self.name + 'concat_1', bottom=self.concat_bottom,
            top=self.top[0])
        self.concat_rnn = LayerParameter('Concat',
            name=self.name + 'concat_2', bottom=self.concat_rnn_bottom,
            top=self.top[1])
        self.__layers__.append(self.concat)
        self.__layers__.append(self.concat_rnn)

    def set_depth_names(self, n1x1, n3x3_r, n3x3, n5x5_r, n5x5, npool_pro):
        if n1x1:
            if self.inception_param[0]:
                self.c1x1_rnn.conv.set('name', n1x1)
            else:
                print "[Error] 1x1 does not exist"
                exit(0)
        if n3x3_r:
            if self.inception_param[1]:
                self.c3x3_rnn_r.conv.set('name', n3x3_r)
            else:
                print "[Error] 3x3 reduce does not exist"
                exit(0)
        if n3x3:
            if self.inception_param[2]:
                self.c3x3_rnn.conv.set('name', n3x3)
            else:
                print "[Error] 3x3 does not exist"
                exit(0)
        if n5x5_r:
            if self.inception_param[3]:
                self.c5x5_rnn_r.conv.set('name', n5x5_r)
            else:
                print "[Error] 5x5 reduce does not exist"
                exit(0)
        if n5x5:
            if self.inception_param[4]:
                self.c5x5_rnn.conv.set('name', n5x5)
        if npool_pro:
            if self.inception_param[6]:
                self.pool_pro_rnn.conv.set('name', npool_pro)
            else:
                print "[Error] pool proj does not exist"
                exit(0)

class InceptionRNN_v2(Block):
    def __init__(self, name, bottom, top, inception_param, sum_rnn_param,
                 param, rnn_shapes):
        self.name = name
        self.bottom = bottom
        self.top = top
        self.inception_param = inception_param
        self.sum_rnn_param = sum_rnn_param
        self.param = param
        self.rnn_shapes = rnn_shapes
        self.__layers__ = []
        self.change()

    @staticmethod
    def no_bias(cp):
        ans = copy.deepcopy(cp)
        ans.set('bias_term', False)
        return ans

    def change(self):
        # check param
        if not (isinstance(self.inception_param, list)
                and len(self.inception_param) == 7):
            print "[Error]: InceptionRNN block receive seven inception_params."
            exit(0)
        self.__layers__ = []

        self.concat_bottom = []
        if self.inception_param[0]:
            self.c1x1_rnn = ConvRNN_v2(self.name+'/c1x1_rnn', self.bottom,
                    self.name+'/c1x1_rnn',
                    [self.inception_param[0],
                     InceptionRNN_v2.no_bias(self.inception_param[0])],
                    self.sum_rnn_param, self.param,
                    self.rnn_shapes[0])
            self.__layers__ += self.c1x1_rnn.__layers__
            self.concat_bottom.append(self.name + '/c1x1_rnn')
        else:
            print "Info: 1x1 conv not set"
        bool_temp = bool(self.inception_param[1]) + bool(self.inception_param[2])
        if bool_temp == 1:
            print "Error: c3x3_r_param and c3x3_param \
                    should be set both or not set both"
            exit(0)
        elif bool_temp == 0:
            print "Info: 3x3 conv not set"
        else:
            self.c3x3_rnn_r = ConvRNN_v2(self.name + '/c3x3_rnn_r', self.bottom,
                     self.name + '/c3x3_rnn_r',
                     [self.inception_param[1],
                      InceptionRNN_v2.no_bias(self.inception_param[1])],
                     self.sum_rnn_param, self.param,
                     self.rnn_shapes[1])
            self.__layers__ += self.c3x3_rnn_r.__layers__
            self.c3x3_rnn = ConvRNN_v2(self.name + '/c3x3_rnn',
                     self.name + '/c3x3_rnn_r',
                     self.name + '/c3x3_rnn',
                     [self.inception_param[2],
                      InceptionRNN_v2.no_bias(self.inception_param[2])],
                     self.sum_rnn_param, self.param,
                     self.rnn_shapes[2])
            self.__layers__ += self.c3x3_rnn.__layers__
            self.concat_bottom.append(self.name + '/c3x3_rnn')
        bool_temp = bool(self.inception_param[3]) + bool(self.inception_param[4])
        if bool_temp == 1:
            print "Error: c5x5_r_param and c5x5_param \
                    should be set both or not set both"
            exit(0)
        elif bool_temp == 0:
            print "Info: 5x5 conv not set"
        else:
            self.c5x5_rnn_r = ConvRNN_v2(self.name + '/c5x5_rnn_r', self.bottom,
                     self.name + '/c5x5_rnn_r',
                     [self.inception_param[3],
                      InceptionRNN_v2.no_bias(self.inception_param[3])],
                     self.sum_rnn_param, self.param,
                     self.rnn_shapes[3])
            self.__layers__ += self.c5x5_rnn_r.__layers__
            self.c5x5_rnn = ConvRNN_v2(self.name + '/c5x5_rnn',
                     self.name + '/c5x5_rnn_r',
                     self.name + '/c5x5_rnn',
                     [self.inception_param[4],
                      InceptionRNN_v2.no_bias(self.inception_param[4])],
                     self.sum_rnn_param, self.param,
                     self.rnn_shapes[4])
            self.__layers__ += self.c5x5_rnn.__layers__
            self.concat_bottom.append(self.name + '/c5x5_rnn')
        if self.inception_param[5]:
            self.pool = LayerParameter('Pooling',
                                       name=self.name + '/pool',
                                       bottom=self.bottom,
                                       top=self.name + '/pool',
                                       pooling_param=self.inception_param[5])
            self.__layers__.append(self.pool)
            self.concat_bottom.append(self.name + '/pool')
            if self.inception_param[6]:
                self.pool_pro_rnn = ConvRNN_v2(self.name + '/pool_pro_rnn',
                    self.name + '/pool',
                    self.name + '/pool_pro',
                    [self.inception_param[6],
                     InceptionRNN_v2.no_bias(self.inception_param[6])],
                    self.sum_rnn_param, self.param,
                    self.rnn_shapes[6])
                self.__layers__ += self.pool_pro_rnn.__layers__
                self.concat_bottom[-1] = self.name + '/pool_pro'
        elif self.inception_param[6]:
            print "pool_pro exists only if pool exists!"
            exit(0)
        if len(self.concat_bottom) <= 1:
            print "Error: concat_bottom should be more than 1"
            traceback.print_stack()
            exit(0)
        self.concat = LayerParameter('Concat',
                                     name=self.name + 'concat_1', bottom=self.concat_bottom,
                                     top=self.top)
        self.__layers__.append(self.concat)

    def set_depth_names(self, n1x1, n3x3_r, n3x3, n5x5_r, n5x5, npool_pro):
        if n1x1:
            if self.inception_param[0]:
                self.c1x1_rnn.conv.set('name', n1x1)
            else:
                print "[Error] 1x1 does not exist"
                exit(0)
        if n3x3_r:
            if self.inception_param[1]:
                self.c3x3_rnn_r.conv.set('name', n3x3_r)
            else:
                print "[Error] 3x3 reduce does not exist"
                exit(0)
        if n3x3:
            if self.inception_param[2]:
                self.c3x3_rnn.conv.set('name', n3x3)
            else:
                print "[Error] 3x3 does not exist"
                exit(0)
        if n5x5_r:
            if self.inception_param[3]:
                self.c5x5_rnn_r.conv.set('name', n5x5_r)
            else:
                print "[Error] 5x5 reduce does not exist"
                exit(0)
        if n5x5:
            if self.inception_param[4]:
                self.c5x5_rnn.conv.set('name', n5x5)
        if npool_pro:
            if self.inception_param[6]:
                self.pool_pro_rnn.conv.set('name', npool_pro)
            else:
                print "[Error] pool proj does not exist"

    @staticmethod
    def get_rnn_shapes(height, width, channels, num_output):
        ans = []
        for c in channels:
            ans.append(BlobShape(num_output, c, height, width))
        return ans

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


filler_constant_0_2 = FillerParameter('constant', value=0.2)
filler_xavier = FillerParameter('xavier')


def get_cp(num_output, pad, kernel_size, stride, bias_term=True, group=0,
           weight_filler=filler_xavier,
           bias_filler=filler_constant_0_2):
    ans = ConvolutionParam(num_output, pad, kernel_size, stride,
                            weight_filler=weight_filler,
                            bias_filler=bias_filler)
    if group != 0:
        ans.set('group', group)
    if not bias_term:
        ans.set('bias_term', False)
    return ans


def get_ipp(num_output, bias_term=True, weight_filler=filler_xavier,
            bias_filler=filler_constant_0_2):
    ans = InnerProductParameter(num_output,
                                 weight_filler=weight_filler,
                                 bias_filler=bias_filler)
    if not bias_term:
        ans.set('bias_term', False)
    return ans

if __name__ == '__main__':
    elt_param = EltwiseParameter(Enum('SUM'), coeff=[0.5, 0.5])
    param = [param_1_1, param_2_0]
    rnn_shape = BlobShape(96, 96, 56, 56)
    rnn_shape2 = BlobShape(96, 96, 1, 1)
    conv_rnn = ConvRNN_v2('conv_rnn', 'data1', 'conv_rnn',
                          [get_cp(96, 1, 3, 1), get_cp(96, 1, 3, 1)],
                          elt_param, param, rnn_shape)
    print display(conv_rnn.__layers__)
    ip_rnn = IPRNN_v2('ip_rnn', 'data1', 'ip_rnn',
                          [get_ipp(96), get_ipp(96)],
                          elt_param, param, rnn_shape2)
    print display(ip_rnn.__layers__)
