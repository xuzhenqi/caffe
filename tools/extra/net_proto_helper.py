import pdb
import copy

# When enable editing a param, taking the reference copy of python

class Compound:
    def initial(self, kwargs):
        for key in self.__titles__:
            # self.name = []
            # if 'name' in kwargs.keys(): self.name = kwargs['name']
            exec("self." + key + " = []");
            exec("if '" + key + "' in kwargs.keys(): self." + key +
                 " = copy.deepcopy(kwargs['" + key + "'])")

    def display(self, level=0):
        #import ipdb;ipdb.set_trace()
        string = " {\n"
        level += 1
        for key in self.__titles__:
            #for v in self.name:
            #    string += v.display(level)
            exec("for v in self." + key + 
            ":\n  if not isinstance(v, Atom):\n    string += level * '  ' + '" 
                    + key + "'\n  string += v.display(level)")
        string += (level - 1) * "  " + "}\n"
        return string

class ParamSpec(Compound):
    def __init__(self, **kwargs):
        self.__titles__ = ['name', 'share_mode', 'lr_mult', 'decay_mult']
        self.initial(kwargs)

class BlobProto(Compound):
    def __init__(self, **kwargs):
        self.__titles__ = ['shape', 'data', 'diff', 'num', 'channels',
                           'height', 'width']
        self.initial(kwargs)


class BlobShape(Compound):
    def __init__(self, **kwargs):
        self.__titles__ = ['dim']
        self.initial(kwargs)


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
    def __init__(self, *kwargs):
        self.__titles__ = ['axis', 'concat_dim']
        self.initial(kwargs)

class ConvolutionParam(Compound):
    def __init__(self, **kwargs):
        self.__titles__ = ['num_output', 'bias_term', 'pad', 'pad_h', 'pad_w',
                           'kernel_size', 'kernel_h', 'kernel_w', 'group',
                           'stride', 'stride_h', 'stride_w', 'weight_filler',
                           'bias_filler', 'engine']
        self.initial(kwargs)


class FillerParameter(Compound):
    def __init__(self, **kwargs):
        self.__titles__ = ['type', 'value', 'min', 'max', 'mean', 'std',
                           'sparse']
        self.initial(kwargs)


class DataParameter(Compound):
    def __init__(self, **kwargs):
        self.__titles__ = ['source', 'batch_size', 'rand_skip', 'backend',
                           'scale', 'mean_file', 'crop_size', 'mirror',
                           'force_encodec_color']
        self.initial(kwargs)


class DropoutParameter(Compound):
    def __init__(self, **kwargs):
        self.__titles__ = ['dropout_ratio']
        self.initial(kwargs)


class ImageDataParameter(Compound):
    def __init__(self, **kwargs):
        self.__titles__ = ['source', 'batch_size', 'rand_skip', 'shuffle',
                           'new_height', 'new_width', 'is_color', 'scale',
                           'mean_file', 'crop_size', 'mirror', 'root_folder']
        self.initial(kwargs)


class InnerProductParameter(Compound):
    def __init__(self, **kwargs):
        self.__titles__ = ['num_output', 'bias_term', 'weight_filler',
                           'bias_filler', 'axis']


class L2NParameter(Compound):
    def __init__(self, **kwargs):
        self.__titles__ = ['eps']
        self.initial(kwargs)


class LRNParameter(Compound):
    def __init__(self, **kwargs):
        self.__titles__ = ['local_size', 'alpha', 'beta', 'norm_region', 'k']
        self.initial(kwargs)


class LSTMParameter(Compound):
    def __init__(self, **kwargs):
        self.__titles__ = ['num_output', 'clipping_threshold', 'weight_filler',
                           'bias_filler', 'batch_size']
        self.initial(kwargs)


class PoolingParameter(Compound):
    def __init__(self, **kwargs):
        self.__titles__ = ['pool', 'pad', 'pad_h', 'pad_w', 'kernel_size',
                           'kernel_h', 'kernel_w', 'stride', 'stride_h',
                           'kernel_w', 'engine', 'global_pooling']
        self.initial(kwargs)


class ReLUParameter(Compound):
    def __init__(self, **kwargs):
        self.__titles__ = ['negative_slope', 'engine']
        self.initial(kwargs)


class TripletLossParameter(Compound):
    def __init__(self, **kwargs):
        self.__titles__ = ['margin']
        self.initial(kwargs)


class TripletImageDataParameter(Compound):
    def __init__(self, **kwargs):
        self.__titles__ = ['statics']
        self.initial(kwargs)


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

class LayerParameter(Compound):
    def __init__(self, **kwargs):
        self.__titles__ = ['name', 'type', 'bottom', 'top', 'phase',
                           'loss_weight', 'param', 'blobs', 'include',
                           'exclude', 'transform_param', 'loss_param',
                           'concat_param', 'convolution_param', 'data_param',
                           'dropout_param', 'image_data_param',
                           'inner_product_param', 'l2n_param', 'lrn_param'
                           'lstm_param', 'pooling_param', 'relu_param',
                           'triplet_loss_param', 'triplet_image_data_param']
        self.initial(kwargs)


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
    def display(self):
        string = ''
        for v in self.__layers__:
            string += 'layer' + v.display()
        return string

class ConvolutionBlock(Block):
    def __init__(self, name, bottom, top, param):
        self.name = name
        self.bottom = bottom
        self.top = top
        self.param = param
        self.c_p = ConvolutionParam(type = [Atom('type', 'Convolution', True)],
            num_output = [Atom('num_output', param[0])],
            pad = [Atom('pad', param[1])], 
            kernel_size = [Atom('kernel_size', param[2])],
            stride = [Atom('stride', param[3])],
            weight_filler = [wf],
            bias_filler = [bf])
        self.c_l_1 = LayerParameter(type = [Atom('type', 'Convolution', True)],
            param = [pw, pb],
            convolution_param = [c_p])
        self.p = LayerParameter(type = [Atom('type', 'ReLU', True)])
        self.change()
        self.__layers__ = [self.c_l_1, self.p]
    def change(self):
        self.c_l_1.name = [Atom('name', self.name + '/conv', True)]
        self.c_l_1.bottom = [Atom('bottom', self.bottom, True)]
        self.c_l_1.top = [Atom('bottom', self.name + '/conv', True)]
        self.p.name = [Atom('name', 'conv1/relu', True)]
        self.p.bottom = [Atom('bottom', self.name + '/conv', True)]
        self.p.top = [Atom('top', self.top, True)]
    def set_name(self):
        None
    def set_bottom(self):
        None
    def set_top(self):
        None



        
pw = ParamSpec(lr_mult = [Atom('lr_mult', 1)],
        decay_mult = [Atom('decay_mult', 1)])
pb = ParamSpec(lr_mult = [Atom('lr_mult', 2)],
        decay_mult = [Atom('decay_mult', 0)])
wf = FillerParameter(type = [Atom('type', 'xavier', True)],
        std = [Atom('std', 0.1)])
bf = FillerParameter(type = [Atom('type', 'constant', True)],
            std = [Atom('value', 0.2)])


if __name__ == '__main__':
    # image_data_layer
    dataset = "/mnt/dataset2/CASIAWebFace/"
    i_d_p = ImageDataParameter(
            source = [Atom('source', '\"' + dataset + 'filelist_crop.txt\"')],
            batch_size = [Atom('batch_size', '60')],
            root_folder = [Atom('root_folder', '\"' + dataset + 'casia_crop/\"')])
    t_i_d_p = TripletImageDataParameter(
            statics = [Atom('statics', '\"' + dataset + 'identities.txt\"')])
    t_i_d_l = LayerParameter(
            name = [Atom('name', 'tripleData', True)],
            type = [Atom('type', 'TripletImageData', True)],
            top = [Atom('top', 'data_1', True), Atom('top', 'data_2', True), 
                Atom('top', 'data_3', True)],
            image_data_param = [i_d_p],
            triplet_image_data_param = [t_i_d_p])
    c_p = ConvolutionParam(type = [Atom('type', 'Convolution', True)],
        num_output = [Atom('num_output', 64)],
        pad = [Atom('pad', 1)], 
        kernel_size = [Atom('kernel_size', 3)],
        stride = [Atom('stride', 1)],
        weight_filler = [wf],
        bias_filler = [bf])
    c_l_1 = LayerParameter(type = [Atom('type', 'Convolution', True)],
        param = [pw, pb],
        convolution_param = [c_p])
    #print c_l.display()
    cb1 = ConvolutionBlock('conv1', 'data_1', 'conv1', [64, 1, 3, 1])
    p_p = PoolingParameter(pool = [Atom('pool', 'MAX')],
            kernel_size = [Atom('kernel_size', 3)],
            stride = [Atom('stride', 2)])
    p_l = LayerParameter()
