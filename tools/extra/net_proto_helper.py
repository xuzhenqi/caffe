import pdb

# When enable editing a param, taking the reference copy of python

class Compound:
    def initial(self, kwargs):
        for key in self.__titles__:
            # self.name = []
            # if 'name' in kwargs.keys(): self.name = kwargs['name']
            exec("self." + key + " = []");
            exec("if '" + key + "' in kwargs.keys(): self." + key +
                 " = kwargs['" + key + "']")

    def display(self, level=0):
        string = level * "  " + "layer {\n"
        level += 1
        for key in self.__titles__:
            #for v in self.name:
            #    string += v.display(level)
            exec("for v in self." + key + ": string += v.display(level)")
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
    def __init__(self, title, value):
        self.title = title
        self.value = value

    def display(self, level):
        return level * "  " + self.title + ": " + str(self.value) + "\n"


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

if __name__ == '__main__':
    l = Layer(name = [Atom('name','test')], type = [Atom('type', 'Convolution')])
    print l.display()


