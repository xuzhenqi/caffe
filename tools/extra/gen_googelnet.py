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
                                   32, '/mnt/dataset3/small/UCF-101/')
    idp_test = ImageDataParameter('/mnt/dataset3/small/source_01_test.txt',
                                  16, '/mnt/dataset3/small/UCF-101/')
    idrnnp = ImageDataRNNParameter(5)
    include_train = NetState(phase=Enum('TRAIN'))
    include_test = NetState(phase=Enum('TEST'))
    transform_train = TransformationParameter(mirror=True, crop_size=224,
                                              mean_value=[104, 117, 123])
    transform_test = TransformationParameter(mirror=False, crop_size=224,
                                             mean_value=[104, 117, 123])
    ep_0_5 = EltwiseParameter(Enum('SUM'), coeff=[0.5, 0.5])
    lrn_param_5_0_0001_0_75 = LRNParameter(5, 0.0001, 0.75)
    pooling_param_max_3_2 = PoolingParameter(Enum('MAX'), 3, 2)
    pooling_param_max_3_1_1 = PoolingParameter(Enum('MAX'), 3, 1, 1)
    pooling_param_ave_5_3 = PoolingParameter(Enum('AVE'), 5, 3)
    pooling_param_ave_7_1 = PoolingParameter(Enum('AVE'), 7, 1)
    drop_param_0_7 = DropoutParameter(0.7)
    drop_param_0_4 = DropoutParameter(0.4)
    acc_param_5 = AccuracyParameter(top_k=5)

    param_0_1 = ParamSpec(0, 1)
    param_0_0 = ParamSpec(0, 0)

    param_0_2_1 = ParamSpec(0.2, 1)
    param_0_4_0 = ParamSpec(0.4, 0)



    params = [[param_0_2_1, param_0_4_0], [param_1_1, param_2_0]]



    layers = []

    # Image Data
    image_data_train = LayerParameter('ImageDataTwo',
                                      name='image_data',
                                      top=['data1', 'data2', 'label'],
                                      image_data_param=idp_train,
                                      image_data_rnn_param=idrnnp,
                                      include=include_train,
                                      transform_param=transform_train)
    image_data_test = LayerParameter('ImageDataTwo',
                                     name='image_data',
                                     top=['data1', 'data2', 'label'],
                                     image_data_param=idp_test,
                                     image_data_rnn_param=idrnnp,
                                     include=include_test,
                                     transform_param=transform_test)
    layers += [image_data_train, image_data_test]

    conv_rnn1 = ConvRNN('conv_rnn1', ['data1', 'data2'],
                    ['conv_rnn1_1', 'conv_rnn1_2'],
                    [get_cp(64, 3, 7, 2), get_cp(64, 1, 3, 1)], ep_0_5, params)
    conv_rnn1.conv.set('name', 'conv1/7x7_s2')
    layers += conv_rnn1.__layers__

    pool1_1 = LayerParameter('Pooling',
                name='pool1_1', bottom='conv_rnn1_1', top='pool1_1',
                pooling_param=pooling_param_max_3_2)
    pool1_2 = LayerParameter('Pooling',
                             name='pool1_2', bottom='conv_rnn1_2',
                             top='pool1_2',
                             pooling_param=pooling_param_max_3_2)
    lrn1_1 = LayerParameter('LRN',
                name='lrn1_1', bottom='pool1_1', top='lrn1_1',
                lrn_param=lrn_param_5_0_0001_0_75)
    lrn1_2 = LayerParameter('LRN',
                            name='lrn1_2', bottom='pool1_2', top='lrn1_2',
                            lrn_param=lrn_param_5_0_0001_0_75)
    layers += [pool1_1, pool1_2, lrn1_1, lrn1_2]

    conv_rnn2_r = ConvRNN('conv_rnn2_r', ['lrn1_1', 'lrn1_2'],
                ['conv_rnn2_r_1', 'conv_rnn2_r_2'],
                [get_cp(64, 0, 1, 1), get_cp(64, 0, 1, 1)], ep_0_5, params)
    conv_rnn2_r.conv.set('name', 'conv2/3x3_reduce')
    conv_rnn2 = ConvRNN('conv_rnn2', ['conv_rnn2_r_1', 'conv_rnn2_r_2'],
                ['conv_rnn2_1', 'conv_rnn2_2'],
                [get_cp(192, 1, 3, 1), get_cp(192, 1, 3, 1)],
                ep_0_5, params)
    conv_rnn2.conv.set('name', 'conv2/3x3')
    layers += conv_rnn2_r.__layers__ + conv_rnn2.__layers__

    lrn2_1 = LayerParameter('LRN',
                name='lrn2_1',
                bottom='conv_rnn2_1',
                top='lrn2_1',
                lrn_param=lrn_param_5_0_0001_0_75)
    lrn2_2 = LayerParameter('LRN',
                            name='lrn2_2',
                            bottom='conv_rnn2_2',
                            top='lrn2_2',
                            lrn_param=lrn_param_5_0_0001_0_75)
    pool2_1 = LayerParameter('Pooling',
                             name='pool2_1',
                             bottom='lrn2_1',
                             top='pool2_1',
                             pooling_param=pooling_param_max_3_2)
    pool2_2 = LayerParameter('Pooling',
                             name='pool2_2',
                             bottom='lrn2_2',
                             top='pool2_2',
                             pooling_param=pooling_param_max_3_2)
    layers += [lrn2_1, lrn2_2, pool2_1, pool2_2]

    inception_rnn3a = InceptionRNN('inception_rnn3a', ['pool2_1', 'pool2_2'],
            ['inception_rnn3a_1', 'inception_rnn3a_2'],
            [get_cp(64, 0, 1, 1), get_cp(96, 0, 1, 1), get_cp(128, 1, 3, 1),
             get_cp(16, 0, 1, 1), get_cp(32, 2, 5, 1),
             pooling_param_max_3_1_1, get_cp(32, 0, 1, 1)],
            ep_0_5, params)
    inception_rnn3a.set_depth_names('inception_3a/1x1',
            'inception_3a/3x3_reduce', 'inception_3a/3x3',
            'inception_3a/5x5_reduce', 'inception_3a/5x5',
            'inception_3a/pool_proj')
    layers += inception_rnn3a.__layers__

    inception_rnn3b = InceptionRNN('inception_rnn3b',
            ['inception_rnn3a_1', 'inception_rnn3a_2'],
            ['inception_rnn3b_1', 'inception_rnn3b_2'],
            [get_cp(128, 0, 1, 1), get_cp(128, 0, 1, 1), get_cp(192, 1, 3, 1),
            get_cp(32, 0, 1, 1), get_cp(96, 2, 5, 1),
            pooling_param_max_3_1_1, get_cp(64, 0, 1, 1)],
            ep_0_5, params)
    inception_rnn3b.set_depth_names('inception_3b/1x1',
                                    'inception_3b/3x3_reduce', 'inception_3b/3x3',
                                    'inception_3b/5x5_reduce', 'inception_3b/5x5',
                                    'inception_3b/pool_proj')
    layers += inception_rnn3b.__layers__

    pool3_1 = LayerParameter('Pooling',
                            name='pool3_1',
                            bottom='inception_rnn3b_1',
                            top='pool3_1',
                            pooling_param=pooling_param_max_3_2)
    pool3_2 = LayerParameter('Pooling',
                             name='pool3_2',
                             bottom='inception_rnn3b_2',
                             top='pool3_2',
                             pooling_param=pooling_param_max_3_2)
    layers += [pool3_1, pool3_2]

    inception_rnn4a = InceptionRNN('inception_rnn4a',
            ['pool3_1', 'pool3_2'],
            ['inception_rnn4a_1', 'inception_rnn4a_2'],
            [get_cp(192, 0, 1, 1), get_cp(96, 0, 1, 1), get_cp(208, 1, 3, 1),
             get_cp(16, 0, 1, 1), get_cp(48, 2, 5, 1),
             pooling_param_max_3_1_1, get_cp(64, 0, 1, 1)],
            ep_0_5, params)
    inception_rnn4a.set_depth_names('inception_4a/1x1',
                                    'inception_4a/3x3_reduce', 'inception_4a/3x3',
                                    'inception_4a/5x5_reduce', 'inception_4a/5x5',
                                    'inception_4a/pool_proj')
    layers += inception_rnn4a.__layers__

    loss1_pool_1 = LayerParameter('Pooling',
                                name='loss1_pool_1',
                                bottom='inception_rnn4a_1',
                                top='loss1_pool_1',
                                pooling_param=pooling_param_ave_5_3)
    loss1_pool_2 = LayerParameter('Pooling',
                                  name='loss1_pool_2',
                                  bottom='inception_rnn4a_2',
                                  top='loss1_pool_2',
                                  pooling_param=pooling_param_ave_5_3)
    layers += [loss1_pool_1, loss1_pool_2]

    loss1_conv_rnn = ConvRNN('loss1_conv_rnn', ['loss1_pool_1', 'loss1_pool_2'],
                        ['loss1_conv_rnn_1', 'loss1_conv_rnn_2'],
                        [get_cp(128, 0, 1, 1), get_cp(128, 0, 1, 1)],
                        ep_0_5, params)
    loss1_conv_rnn.conv.set('name', 'loss1/conv')
    layers += loss1_conv_rnn.__layers__

    loss1_fcrnn = IPRNN('loss1_fcrnn', ['loss1_conv_rnn_1', 'loss1_conv_rnn_2'],
                        ['loss1_fcrnn_1', 'loss1_fcrnn_2'],
                        [get_ipp(1024), get_ipp(1024)], ep_0_5, params)
    loss1_fcrnn.ip.set('name', 'loss1/fc')
    layers += loss1_fcrnn.__layers__

    loss1_drop_1 = LayerParameter('Dropout',
                                name='loss1_drop_1',
                                bottom='loss1_fcrnn_1',
                                top='loss1_drop_1',
                                dropout_param=drop_param_0_7)
    loss1_drop_2 = LayerParameter('Dropout',
                                  name='loss1_drop_2',
                                  bottom='loss1_fcrnn_2',
                                  top='loss1_drop_2',
                                  dropout_param=drop_param_0_7)
    loss1_class = LayerParameter('InnerProduct',
                                   name='loss1/classifier101',
                                   bottom=['loss1_drop_1', 'loss1_drop_2'],
                                   top=['loss1_class_1', 'loss1_class_2'],
                                   inner_product_param=get_ipp(101),
                                   param=params[0])
    loss1_loss_1 = LayerParameter('SoftmaxWithLoss',
                                  name='loss1_loss_1',
                                  bottom=['loss1_class_1', 'label'],
                                  top='loss1_loss_1',
                                  loss_weight=0.3)
    loss1_loss_2 = LayerParameter('SoftmaxWithLoss',
                                  name='loss1_loss_2',
                                  bottom=['loss1_class_2', 'label'],
                                  top='loss1_loss_2',
                                  loss_weight=0.3)
    loss1_top1_1 = LayerParameter('Accuracy',
                                  name='loss1_top1_1',
                                  bottom=['loss1_class_1', 'label'],
                                  top='loss1_top1_1',
                                  include=include_test)
    loss1_top1_2 = LayerParameter('Accuracy',
                                  name='loss1_top1_2',
                                  bottom=['loss1_class_2', 'label'],
                                  top='loss1_top1_2',
                                  include=include_test)
    loss1_top5_1 = LayerParameter('Accuracy',
                                  name='loss1_top5_1',
                                  bottom=['loss1_class_1', 'label'],
                                  top='loss1_top5_1',
                                  include=include_test,
                                  accuracy_param=acc_param_5)
    loss1_top5_2 = LayerParameter('Accuracy',
                                  name='loss1_top5_2',
                                  bottom=['loss1_class_2', 'label'],
                                  top='loss1_top5_2',
                                  include=include_test,
                                  accuracy_param=acc_param_5)
    layers += [loss1_drop_1, loss1_drop_2, loss1_class,
               loss1_loss_1, loss1_loss_2, loss1_top1_1, loss1_top1_2,
               loss1_top5_1, loss1_top5_2]

    inception_rnn4b = InceptionRNN('inception_rnn4b',
                                   ['inception_rnn4a_1', 'inception_rnn4a_2'],
                                   ['inception_rnn4b_1', 'inception_rnn4b_2'],
                                   [get_cp(160, 0, 1, 1), get_cp(112, 0, 1, 1), get_cp(224, 1, 3, 1),
                                    get_cp(24, 0, 1, 1), get_cp(64, 2, 5, 1),
                                    pooling_param_max_3_1_1, get_cp(64, 0, 1, 1)],
                                   ep_0_5, params)
    inception_rnn4b.set_depth_names('inception_4b/1x1',
                                    'inception_4b/3x3_reduce', 'inception_4b/3x3',
                                    'inception_4b/5x5_reduce', 'inception_4b/5x5',
                                    'inception_4b/pool_proj')
    layers += inception_rnn4b.__layers__
    inception_rnn4c = InceptionRNN('inception_rnn4c',
                                   ['inception_rnn4b_1', 'inception_rnn4b_2'],
                                   ['inception_rnn4c_1', 'inception_rnn4c_2'],
                                   [get_cp(128, 0, 1, 1), get_cp(128, 0, 1, 1), get_cp(256, 1, 3, 1),
                                    get_cp(24, 0, 1, 1), get_cp(64, 2, 5, 1),
                                    pooling_param_max_3_1_1, get_cp(64, 0, 1, 1)],
                                   ep_0_5, params)
    inception_rnn4c.set_depth_names('inception_4c/1x1',
                                    'inception_4c/3x3_reduce', 'inception_4c/3x3',
                                    'inception_4c/5x5_reduce', 'inception_4c/5x5',
                                    'inception_4c/pool_proj')
    layers += inception_rnn4c.__layers__

    inception_rnn4d = InceptionRNN('inception_rnn4d',
                                   ['inception_rnn4c_1', 'inception_rnn4c_2'],
                                   ['inception_rnn4d_1', 'inception_rnn4d_2'],
                                   [get_cp(112, 0, 1, 1), get_cp(144, 0, 1, 1), get_cp(288, 1, 3, 1),
                                    get_cp(32, 0, 1, 1), get_cp(64, 2, 5, 1),
                                    pooling_param_max_3_1_1, get_cp(64, 0, 1, 1)],
                                   ep_0_5, params)
    inception_rnn4d.set_depth_names('inception_4d/1x1',
                                    'inception_4d/3x3_reduce', 'inception_4d/3x3',
                                    'inception_4d/5x5_reduce', 'inception_4d/5x5',
                                    'inception_4d/pool_proj')
    layers += inception_rnn4d.__layers__

    loss2_pool_1 = LayerParameter('Pooling',
                                  name='loss2_pool_1',
                                  bottom='inception_rnn4d_1',
                                  top='loss2_pool_1',
                                  pooling_param=pooling_param_ave_5_3)
    loss2_pool_2 = LayerParameter('Pooling',
                                  name='losss2_pool_2',
                                  bottom='inception_rnn4d_2',
                                  top='loss2_pool_2',
                                  pooling_param=pooling_param_ave_5_3)
    layers += [loss2_pool_1, loss2_pool_2]

    loss2_conv_rnn = ConvRNN('loss2_conv_rnn', ['loss2_pool_1', 'loss2_pool_2'],
                             ['loss2_conv_rnn_1', 'loss2_conv_rnn_2'],
                             [get_cp(128, 0, 1, 1), get_cp(128, 0, 1, 1)],
                             ep_0_5, params)
    loss2_conv_rnn.conv.set('name', 'loss2/conv')
    layers += loss2_conv_rnn.__layers__

    loss2_fcrnn = IPRNN('loss2_fcrnn', ['loss2_conv_rnn_1', 'loss2_conv_rnn_2'],
                        ['loss2_fcrnn_1', 'loss2_fcrnn_2'],
                        [get_ipp(1024), get_ipp(1024)], ep_0_5, params)
    loss2_fcrnn.ip.set('name', 'loss2/fc')
    layers += loss2_fcrnn.__layers__

    loss2_drop_1 = LayerParameter('Dropout',
                                  name='loss2_drop_1',
                                  bottom='loss2_fcrnn_1',
                                  top='loss2_drop_1',
                                  dropout_param=drop_param_0_7)
    loss2_drop_2 = LayerParameter('Dropout',
                                  name='loss2_drop_2',
                                  bottom='loss2_fcrnn_2',
                                  top='loss2_drop_2',
                                  dropout_param=drop_param_0_7)
    loss2_class = LayerParameter('InnerProduct',
                                   name='loss2/classifier101',
                                   bottom=['loss2_drop_1', 'loss2_drop_2'],
                                   top=['loss2_class_1', 'loss2_class_2'],
                                   inner_product_param=get_ipp(101),
                                   param=params[0])
    loss2_loss_1 = LayerParameter('SoftmaxWithLoss',
                                  name='loss2_loss_1',
                                  bottom=['loss2_class_1', 'label'],
                                  top='loss2_loss_1',
                                  loss_weight=0.3)
    loss2_loss_2 = LayerParameter('SoftmaxWithLoss',
                                  name='loss2_loss_2',
                                  bottom=['loss2_class_2', 'label'],
                                  top='loss2_loss_2',
                                  loss_weight=0.3)
    loss2_top1_1 = LayerParameter('Accuracy',
                                  name='loss2_top1_1',
                                  bottom=['loss2_class_1', 'label'],
                                  top='loss2_top1_1',
                                  include=include_test)
    loss2_top1_2 = LayerParameter('Accuracy',
                                  name='loss2_top1_2',
                                  bottom=['loss2_class_2', 'label'],
                                  top='loss2_top1_2',
                                  include=include_test)
    loss2_top5_1 = LayerParameter('Accuracy',
                                  name='loss2_top5_1',
                                  bottom=['loss2_class_1', 'label'],
                                  top='loss2_top5_1',
                                  include=include_test,
                                  accuracy_param=acc_param_5)
    loss2_top5_2 = LayerParameter('Accuracy',
                                  name='loss2_top5_2',
                                  bottom=['loss2_class_2', 'label'],
                                  top='loss2_top5_2',
                                  include=include_test,
                                  accuracy_param=acc_param_5)
    layers += [loss2_drop_1, loss2_drop_2, loss2_class,
               loss2_loss_1, loss2_loss_2, loss2_top1_1, loss2_top1_2,
               loss2_top5_1, loss2_top5_2]

    inception_rnn4e = InceptionRNN('inception_rnn4e',
                                   ['inception_rnn4d_1', 'inception_rnn4d_2'],
                                   ['inception_rnn4e_1', 'inception_rnn4e_2'],
                                   [get_cp(256, 0, 1, 1), get_cp(160, 0, 1, 1), get_cp(320, 1, 3, 1),
                                    get_cp(32, 0, 1, 1), get_cp(128, 2, 5, 1),
                                    pooling_param_max_3_1_1, get_cp(128, 0, 1, 1)],
                                   ep_0_5, params)
    inception_rnn4e.set_depth_names('inception_4e/1x1',
                                    'inception_4e/3x3_reduce', 'inception_4e/3x3',
                                    'inception_4e/5x5_reduce', 'inception_4e/5x5',
                                    'inception_4e/pool_proj')
    layers += inception_rnn4e.__layers__

    pool4_1 = LayerParameter('Pooling',
                                  name='pool4_1',
                                  bottom='inception_rnn4e_1',
                                  top='pool4_1',
                                  pooling_param=pooling_param_max_3_2)
    pool4_2 = LayerParameter('Pooling',
                                  name='pool4_2',
                                  bottom='inception_rnn4e_2',
                                  top='pool4_2',
                                  pooling_param=pooling_param_max_3_2)
    layers += [pool4_1, pool4_2]

    inception_rnn5a = InceptionRNN('inception_rnn5a',
                                   ['pool4_1', 'pool4_2'],
                                   ['inception_rnn5a_1', 'inception_rnn5a_2'],
                                   [get_cp(256, 0, 1, 1), get_cp(160, 0, 1, 1), get_cp(320, 1, 3, 1),
                                    get_cp(32, 0, 1, 1), get_cp(128, 2, 5, 1),
                                    pooling_param_max_3_1_1, get_cp(128, 0, 1, 1)],
                                   ep_0_5, params)
    inception_rnn5a.set_depth_names('inception_5a/1x1',
                                    'inception_5a/3x3_reduce', 'inception_5a/3x3',
                                    'inception_5a/5x5_reduce', 'inception_5a/5x5',
                                    'inception_5a/pool_proj')
    layers += inception_rnn5a.__layers__

    inception_rnn5b = InceptionRNN('inception_rnn5b',
                                   ['inception_rnn5a_1', 'inception_rnn5a_2'],
                                   ['inception_rnn5b_1', 'inception_rnn5b_2'],
                                   [get_cp(384, 0, 1, 1), get_cp(192, 0, 1, 1), get_cp(384, 1, 3, 1),
                                    get_cp(48, 0, 1, 1), get_cp(128, 2, 5, 1),
                                    pooling_param_max_3_1_1, get_cp(128, 0, 1, 1)],
                                   ep_0_5, params)
    inception_rnn5b.set_depth_names('inception_5b/1x1',
                                    'inception_5b/3x3_reduce', 'inception_5b/3x3',
                                    'inception_5b/5x5_reduce', 'inception_5b/5x5',
                                    'inception_5b/pool_proj')
    layers += inception_rnn5b.__layers__

    pool5_1 = LayerParameter('Pooling',
                             name='pool5_1',
                             bottom='inception_rnn5b_1',
                             top='pool5_1',
                             pooling_param=pooling_param_ave_7_1)
    pool5_2 = LayerParameter('Pooling',
                             name='pool5_2',
                             bottom='inception_rnn5b_2',
                             top='pool5_2',
                             pooling_param=pooling_param_ave_7_1)
    layers += [pool5_1, pool5_2]

    loss3_drop_1 = LayerParameter('Dropout',
                                  name='loss3_drop_1',
                                  bottom='pool5_1',
                                  top='loss3_drop_1',
                                  dropout_param=drop_param_0_4)
    loss3_drop_2 = LayerParameter('Dropout',
                                  name='loss3_drop_2',
                                  bottom='pool5_2',
                                  top='loss3_drop_2',
                                  dropout_param=drop_param_0_4)
    layers += [loss3_drop_1, loss3_drop_2]

    loss3_fcrnn = IPRNN('loss3_fcrnn', ['loss3_drop_1', 'loss3_drop_2'],
                        ['loss3_fcrnn_1', 'loss3_fcrnn_2'],
                        [get_ipp(1024), get_ipp(1024)], ep_0_5, params)
    loss3_fcrnn.ip.set('name', 'loss3/classifier1')
    layers += loss3_fcrnn.__layers__

    loss3_class = LayerParameter('InnerProduct',
                                   name='loss3/classifier101',
                                   bottom=['loss3_fcrnn_1', 'loss3_fcrnn_2'],
                                   top=['loss3_class_1', 'loss3_class_2'],
                                   inner_product_param=get_ipp(101),
                                   param=params[0])
    loss3_loss_1 = LayerParameter('SoftmaxWithLoss',
                                  name='loss3_loss_1',
                                  bottom=['loss3_class_1', 'label'],
                                  top='loss3_loss_1')
    loss3_loss_2 = LayerParameter('SoftmaxWithLoss',
                                  name='loss3_loss_2',
                                  bottom=['loss3_class_2', 'label'],
                                  top='loss3_loss_2')
    loss3_top1_1 = LayerParameter('Accuracy',
                                  name='loss3_top1_1',
                                  bottom=['loss3_class_1', 'label'],
                                  top='loss3_top1_1',
                                  include=include_test)
    loss3_top1_2 = LayerParameter('Accuracy',
                                  name='loss3_top1_2',
                                  bottom=['loss3_class_2', 'label'],
                                  top='loss3_top1_2',
                                  include=include_test)
    loss3_top5_1 = LayerParameter('Accuracy',
                                  name='loss3_top5_1',
                                  bottom=['loss3_class_1', 'label'],
                                  top='loss3_top5_1',
                                  include=include_test,
                                  accuracy_param=acc_param_5)
    loss3_top5_2 = LayerParameter('Accuracy',
                                  name='loss3_top5_2',
                                  bottom=['loss3_class_2', 'label'],
                                  top='loss3_top5_2',
                                  include=include_test,
                                  accuracy_param=acc_param_5)
    layers += [loss3_class,
               loss3_loss_1, loss3_loss_2, loss3_top1_1, loss3_top1_2,
               loss3_top5_1, loss3_top5_2]
    print display(layers)



