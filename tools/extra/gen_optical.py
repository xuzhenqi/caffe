import sys
CAFFE_ROOT = '/home/xuzhenqi/research/caffe/'
NET_PROTO_HELPER_ROOT = CAFFE_ROOT + 'tools/extra/'
sys.path.append(NET_PROTO_HELPER_ROOT)
from net_proto_helper import *
import argparse

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('-f', '--freeze', type=bool, default=False,
                    help='Whether to freeze the depth param')
args = parser.parse_args()

if __name__ == '__main__':
    idp_train = ImageDataParameter('/mnt/dataset3/small/source_01_train.txt',
                                   64, '/mnt/dataset3/small/UCF-101/')
    idp_test = ImageDataParameter('/mnt/dataset3/small/source_01_test.txt',
                                  32, '/mnt/dataset3/small/UCF-101/')
    include_train = NetState(phase=Enum('TRAIN'))
    include_test = NetState(phase=Enum('TEST'))
    transform_train = TransformationParameter(mirror=True, crop_size=224,
                                              mean_value=[104, 117, 123])
    transform_test = TransformationParameter(mirror=False, crop_size=224,
                                             mean_value=[104, 117, 123])
    pooling_param_max_3_2 = PoolingParameter(Enum('MAX'), 3, 2)
    drop_param_0_7 = DropoutParameter(0.7)
    drop_param_0_5 = DropoutParameter(0.5)
    param_0_1 = ParamSpec(0, 1)
    param_0_0 = ParamSpec(0, 0)

    layers = []

    # Image Data
    image_data_train = LayerParameter('ImageData',
                                      name='image_data',
                                      top=['data1', 'label'],
                                      image_data_param=idp_train,
                                      include=include_train,
                                      transform_param=transform_train)
    image_data_test = LayerParameter('ImageData',
                                     name='image_data',
                                     top=['data1', 'label'],
                                     image_data_param=idp_test,
                                     include=include_test,
                                     transform_param=transform_test)
    layers += [image_data_train, image_data_test]

    conv1 = Conv_relu('conv1', 'data', 'conv1', get_cp(96, 3, 7, 3))
    layers += conv1.__layers__

    conv2 = Conv_relu('conv2', 'conv1', 'conv2', get_cp(96, 1, 3, 1))
    layers += conv2.__layers__

    pool2 = LayerParameter('Pooling',
                name='pool2',
                bottom='conv2',
                top='pool2',
                pooling_param=pooling_param_max_3_2)
    layers.append(pool2)

    conv3 = Conv_relu('conv3', 'pool2', 'conv3', get_cp(96, 1, 3, 1))
    layers += conv3.__layers__

    pool3 = LayerParameter('Pooling',
                name='pool3',
                bottom='conv3',
                top='pool3',
                pooling_param=pooling_param_max_3_2)
    layers.append(pool3)

    fc4 = Fully_relu('fc4', 'pool3', 'fc4', get_ipp(1024))
    layers += fc4.__layers__

    drop4 = LayerParameter('Dropout',
                name='drop4',
                bottom='fc4',
                top='fc4',
                dropout_param=drop_param_0_5)
    layers.append(drop4)

    ip5 = LayerParameter('InnerProduct',
               name='ip5',
               bottom='fc4',
               top='ip5',
               inner_product_param=get_ipp(101),
               param=[param_1_1, param_2_0])
    loss = LayerParameter('SoftmaxWithLoss',
               name='loss',
               bottom='ip5',
               top='loss')
    acc = LayerParameter('Accuracy',
               name='acc',
               bottom='ip5',
               top='acc')
    layers += [ip5, loss, acc]
    print display(layers)
