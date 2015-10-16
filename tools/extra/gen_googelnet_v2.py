import sys
from net_proto_helper import *
import argparse

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('-f', '--freeze', type=bool, default=False,
                    help='Whether to freeze the depth param')
args = parser.parse_args()

if __name__ == '__main__':
    batch_size = 32
    idp_train = ImageDataParameter('/mnt/dataset3/small/source_01_train.txt',
                                   batch_size, '/mnt/dataset3/small/UCF-101/')
    idp_test = ImageDataParameter('/mnt/dataset3/small/source_01_test.txt',
                                  batch_size, '/mnt/dataset3/small/UCF-101/')
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
    image_data_train = LayerParameter('ImageDataRNN',
                                      name='image_data',
                                      top=['data', 'label', 'begin_marker'],
                                      image_data_param=idp_train,
                                      image_data_rnn_param=idrnnp,
                                      include=include_train,
                                      transform_param=transform_train)
    image_data_test = LayerParameter('ImageDataRNN',
                                     name='image_data',
                                     top=['data', 'label', 'begin_marker'],
                                     image_data_param=idp_test,
                                     image_data_rnn_param=idrnnp,
                                     include=include_test,
                                     transform_param=transform_test)
    layers += [image_data_train, image_data_test]

    conv_rnn1 = ConvRNN_v2('conv_rnn1', 'data',
                    'conv_rnn1',
                    [get_cp(64, 3, 7, 2), get_cp(64, 1, 3, 1)], ep_0_5,
                    params, BlobShape(batch_size, 64, 112, 112))
    conv_rnn1.conv.set('name', 'conv1/7x7_s2')
    layers += conv_rnn1.__layers__

    pool1_1 = LayerParameter('Pooling',
                name='pool1', bottom='conv_rnn1', top='pool1',
                pooling_param=pooling_param_max_3_2)

    lrn1_1 = LayerParameter('LRN',
                name='lrn1', bottom='pool1', top='lrn1',
                lrn_param=lrn_param_5_0_0001_0_75)
    layers += [pool1_1, lrn1_1]

    conv_rnn2_r = ConvRNN_v2('conv_rnn2_r', 'lrn1',
                'conv_rnn2_r',
                [get_cp(64, 0, 1, 1), get_cp(64, 0, 1, 1)], ep_0_5, params,
                BlobShape(batch_size, 64, 56, 56))
    conv_rnn2_r.conv.set('name', 'conv2/3x3_reduce')
    conv_rnn2 = ConvRNN_v2('conv_rnn2', 'conv_rnn2_r',
                'conv_rnn2',
                [get_cp(192, 1, 3, 1), get_cp(192, 1, 3, 1)],
                ep_0_5, params, BlobShape(batch_size, 192, 56, 56))
    conv_rnn2.conv.set('name', 'conv2/3x3')
    layers += conv_rnn2_r.__layers__ + conv_rnn2.__layers__

    lrn2 = LayerParameter('LRN',
                name='lrn2',
                bottom='conv_rnn2',
                top='lrn2',
                lrn_param=lrn_param_5_0_0001_0_75)
    pool2 = LayerParameter('Pooling',
                             name='pool2',
                             bottom='lrn2',
                             top='pool2',
                             pooling_param=pooling_param_max_3_2)
    layers += [lrn2, pool2]

    inception_rnn3a = InceptionRNN_v2('inception_rnn3a', 'pool2',
            'inception_rnn3a',
            [get_cp(64, 0, 1, 1), get_cp(96, 0, 1, 1), get_cp(128, 1, 3, 1),
             get_cp(16, 0, 1, 1), get_cp(32, 2, 5, 1),
             pooling_param_max_3_1_1, get_cp(32, 0, 1, 1)],
            ep_0_5, params,
            InceptionRNN_v2.get_rnn_shapes(28, 28,
                        [64, 96, 128, 16, 32, 0, 32], batch_size))
    inception_rnn3a.set_depth_names('inception_3a/1x1',
            'inception_3a/3x3_reduce', 'inception_3a/3x3',
            'inception_3a/5x5_reduce', 'inception_3a/5x5',
            'inception_3a/pool_proj')
    layers += inception_rnn3a.__layers__

    inception_rnn3b = InceptionRNN_v2('inception_rnn3b',
            'inception_rnn3a',
            'inception_rnn3b',
            [get_cp(128, 0, 1, 1), get_cp(128, 0, 1, 1), get_cp(192, 1, 3, 1),
            get_cp(32, 0, 1, 1), get_cp(96, 2, 5, 1),
            pooling_param_max_3_1_1, get_cp(64, 0, 1, 1)],
            ep_0_5, params,
            InceptionRNN_v2.get_rnn_shapes(28, 28,
                    [128, 128, 192, 32, 96, 0, 64], batch_size))
    inception_rnn3b.set_depth_names('inception_3b/1x1',
                                    'inception_3b/3x3_reduce', 'inception_3b/3x3',
                                    'inception_3b/5x5_reduce', 'inception_3b/5x5',
                                    'inception_3b/pool_proj')
    layers += inception_rnn3b.__layers__

    pool3 = LayerParameter('Pooling',
                            name='pool3',
                            bottom='inception_rnn3b',
                            top='pool3',
                            pooling_param=pooling_param_max_3_2)
    layers += [pool3]

    inception_rnn4a = InceptionRNN_v2('inception_rnn4a',
            'pool3',
            'inception_rnn4a',
            [get_cp(192, 0, 1, 1), get_cp(96, 0, 1, 1), get_cp(208, 1, 3, 1),
             get_cp(16, 0, 1, 1), get_cp(48, 2, 5, 1),
             pooling_param_max_3_1_1, get_cp(64, 0, 1, 1)],
            ep_0_5, params, InceptionRNN_v2.get_rnn_shapes(14, 14,
                    [192, 96, 208, 16, 48, 0, 64], batch_size))
    inception_rnn4a.set_depth_names('inception_4a/1x1',
                                    'inception_4a/3x3_reduce', 'inception_4a/3x3',
                                    'inception_4a/5x5_reduce', 'inception_4a/5x5',
                                    'inception_4a/pool_proj')
    layers += inception_rnn4a.__layers__

    loss1_pool = LayerParameter('Pooling',
                                name='loss1_pool',
                                bottom='inception_rnn4a',
                                top='loss1_pool',
                                pooling_param=pooling_param_ave_5_3)
    layers += [loss1_pool]

    loss1_conv_rnn = ConvRNN_v2('loss1_conv_rnn', 'loss1_pool',
                        'loss1_conv_rnn',
                        [get_cp(128, 0, 1, 1), get_cp(128, 0, 1, 1)],
                        ep_0_5, params, BlobShape(batch_size, 128, 4, 4))
    loss1_conv_rnn.conv.set('name', 'loss1/conv')
    layers += loss1_conv_rnn.__layers__

    loss1_fcrnn = IPRNN_v2('loss1_fcrnn', 'loss1_conv_rnn',
                        'loss1_fcrnn',
                        [get_ipp(1024), get_ipp(1024)], ep_0_5, params,
                        BlobShape(batch_size, 1024))
    loss1_fcrnn.ip.set('name', 'loss1/fc')
    layers += loss1_fcrnn.__layers__

    loss1_drop = LayerParameter('Dropout',
                                name='loss1_drop',
                                bottom='loss1_fcrnn',
                                top='loss1_drop',
                                dropout_param=drop_param_0_7)
    loss1_class = LayerParameter('InnerProduct',
                                   name='loss1/classifier101',
                                   bottom='loss1_drop',
                                   top='loss1_class',
                                   inner_product_param=get_ipp(101),
                                   param=params[0])
    loss1_loss = LayerParameter('SoftmaxWithLoss',
                                  name='loss1_loss',
                                  bottom=['loss1_class', 'label'],
                                  top='loss1_loss',
                                  loss_weight=0.3)
    loss1_top1 = LayerParameter('Accuracy',
                                  name='loss1_top1',
                                  bottom=['loss1_class', 'label'],
                                  top='loss1_top1',
                                  include=include_test)
    loss1_top5 = LayerParameter('Accuracy',
                                  name='loss1_top5',
                                  bottom=['loss1_class', 'label'],
                                  top='loss1_top5',
                                  include=include_test,
                                  accuracy_param=acc_param_5)
    layers += [loss1_drop, loss1_class,
               loss1_loss, loss1_top1,
               loss1_top5]

    inception_rnn4b = InceptionRNN_v2('inception_rnn4b',
              'inception_rnn4a',
              'inception_rnn4b',
              [get_cp(160, 0, 1, 1), get_cp(112, 0, 1, 1), get_cp(224, 1, 3, 1),
               get_cp(24, 0, 1, 1), get_cp(64, 2, 5, 1),
               pooling_param_max_3_1_1, get_cp(64, 0, 1, 1)],
              ep_0_5, params, InceptionRNN_v2.get_rnn_shapes(14, 14,
                         [160, 112, 224, 24, 64, 0, 64], batch_size))
    inception_rnn4b.set_depth_names('inception_4b/1x1',
                                    'inception_4b/3x3_reduce', 'inception_4b/3x3',
                                    'inception_4b/5x5_reduce', 'inception_4b/5x5',
                                    'inception_4b/pool_proj')
    layers += inception_rnn4b.__layers__
    inception_rnn4c = InceptionRNN_v2('inception_rnn4c',
              'inception_rnn4b',
              'inception_rnn4c',
              [get_cp(128, 0, 1, 1), get_cp(128, 0, 1, 1), get_cp(256, 1, 3, 1),
               get_cp(24, 0, 1, 1), get_cp(64, 2, 5, 1),
               pooling_param_max_3_1_1, get_cp(64, 0, 1, 1)],
              ep_0_5, params, InceptionRNN_v2.get_rnn_shapes(14, 14,
                         [128, 128, 256, 24, 64, 0, 64], batch_size))
    inception_rnn4c.set_depth_names('inception_4c/1x1',
                                    'inception_4c/3x3_reduce', 'inception_4c/3x3',
                                    'inception_4c/5x5_reduce', 'inception_4c/5x5',
                                    'inception_4c/pool_proj')
    layers += inception_rnn4c.__layers__

    inception_rnn4d = InceptionRNN_v2('inception_rnn4d',
              'inception_rnn4c',
              'inception_rnn4d',
              [get_cp(112, 0, 1, 1), get_cp(144, 0, 1, 1), get_cp(288, 1, 3, 1),
               get_cp(32, 0, 1, 1), get_cp(64, 2, 5, 1),
               pooling_param_max_3_1_1, get_cp(64, 0, 1, 1)],
              ep_0_5, params, InceptionRNN_v2.get_rnn_shapes(14, 14,
                     [112, 144, 288, 32, 64, 0, 64], batch_size))
    inception_rnn4d.set_depth_names('inception_4d/1x1',
                                    'inception_4d/3x3_reduce', 'inception_4d/3x3',
                                    'inception_4d/5x5_reduce', 'inception_4d/5x5',
                                    'inception_4d/pool_proj')
    layers += inception_rnn4d.__layers__

    loss2_pool = LayerParameter('Pooling',
                                  name='loss2_pool',
                                  bottom='inception_rnn4d',
                                  top='loss2_pool',
                                  pooling_param=pooling_param_ave_5_3)
    layers += [loss2_pool]

    loss2_conv_rnn = ConvRNN_v2('loss2_conv_rnn', 'loss2_pool',
                             'loss2_conv_rnn',
                             [get_cp(128, 0, 1, 1), get_cp(128, 0, 1, 1)],
                             ep_0_5, params, BlobShape(batch_size, 128, 4, 4))
    loss2_conv_rnn.conv.set('name', 'loss2/conv')
    layers += loss2_conv_rnn.__layers__

    loss2_fcrnn = IPRNN_v2('loss2_fcrnn', 'loss2_conv_rnn',
                        'loss2_fcrnn',
                        [get_ipp(1024), get_ipp(1024)], ep_0_5, params,
                        BlobShape(batch_size, 1024))
    loss2_fcrnn.ip.set('name', 'loss2/fc')
    layers += loss2_fcrnn.__layers__

    loss2_drop = LayerParameter('Dropout',
                                  name='loss2_drop',
                                  bottom='loss2_fcrnn',
                                  top='loss2_drop',
                                  dropout_param=drop_param_0_7)
    loss2_class = LayerParameter('InnerProduct',
                                   name='loss2/classifier101',
                                   bottom='loss2_drop',
                                   top='loss2_class',
                                   inner_product_param=get_ipp(101),
                                   param=params[0])
    loss2_loss = LayerParameter('SoftmaxWithLoss',
                                  name='loss2_loss',
                                  bottom=['loss2_class', 'label'],
                                  top='loss2_loss',
                                  loss_weight=0.3)
    loss2_top1 = LayerParameter('Accuracy',
                                  name='loss2_top1',
                                  bottom=['loss2_class', 'label'],
                                  top='loss2_top1',
                                  include=include_test)
    loss2_top5 = LayerParameter('Accuracy',
                                  name='loss2_top5',
                                  bottom=['loss2_class', 'label'],
                                  top='loss2_top5',
                                  include=include_test,
                                  accuracy_param=acc_param_5)
    layers += [loss2_drop, loss2_class,
               loss2_loss, loss2_top1,
               loss2_top5]

    inception_rnn4e = InceptionRNN_v2('inception_rnn4e',
              'inception_rnn4d',
              'inception_rnn4e',
              [get_cp(256, 0, 1, 1), get_cp(160, 0, 1, 1), get_cp(320, 1, 3, 1),
               get_cp(32, 0, 1, 1), get_cp(128, 2, 5, 1),
               pooling_param_max_3_1_1, get_cp(128, 0, 1, 1)],
              ep_0_5, params, InceptionRNN_v2.get_rnn_shapes(14, 14,
                    [256, 160, 320, 32, 128, 0, 128], batch_size))
    inception_rnn4e.set_depth_names('inception_4e/1x1',
                                    'inception_4e/3x3_reduce', 'inception_4e/3x3',
                                    'inception_4e/5x5_reduce', 'inception_4e/5x5',
                                    'inception_4e/pool_proj')
    layers += inception_rnn4e.__layers__

    pool4 = LayerParameter('Pooling',
                                  name='pool4',
                                  bottom='inception_rnn4e',
                                  top='pool4',
                                  pooling_param=pooling_param_max_3_2)
    layers += [pool4]

    inception_rnn5a = InceptionRNN_v2('inception_rnn5a',
              'pool4',
              'inception_rnn5a',
              [get_cp(256, 0, 1, 1), get_cp(160, 0, 1, 1), get_cp(320, 1, 3, 1),
               get_cp(32, 0, 1, 1), get_cp(128, 2, 5, 1),
               pooling_param_max_3_1_1, get_cp(128, 0, 1, 1)],
              ep_0_5, params, InceptionRNN_v2.get_rnn_shapes(7, 7,
                     [256, 160, 320, 32, 128, 0, 128], batch_size))
    inception_rnn5a.set_depth_names('inception_5a/1x1',
                                    'inception_5a/3x3_reduce', 'inception_5a/3x3',
                                    'inception_5a/5x5_reduce', 'inception_5a/5x5',
                                    'inception_5a/pool_proj')
    layers += inception_rnn5a.__layers__

    inception_rnn5b = InceptionRNN_v2('inception_rnn5b',
              'inception_rnn5a',
              'inception_rnn5b',
              [get_cp(384, 0, 1, 1), get_cp(192, 0, 1, 1), get_cp(384, 1, 3, 1),
               get_cp(48, 0, 1, 1), get_cp(128, 2, 5, 1),
               pooling_param_max_3_1_1, get_cp(128, 0, 1, 1)],
              ep_0_5, params, InceptionRNN_v2.get_rnn_shapes(7, 7,
                     [384, 192, 384, 48, 128, 0, 128], batch_size))
    inception_rnn5b.set_depth_names('inception_5b/1x1',
                                    'inception_5b/3x3_reduce', 'inception_5b/3x3',
                                    'inception_5b/5x5_reduce', 'inception_5b/5x5',
                                    'inception_5b/pool_proj')
    layers += inception_rnn5b.__layers__

    pool5 = LayerParameter('Pooling',
                             name='pool5',
                             bottom='inception_rnn5b',
                             top='pool5',
                             pooling_param=pooling_param_ave_7_1)
    layers += [pool5]

    loss3_drop = LayerParameter('Dropout',
                                  name='loss3_drop',
                                  bottom='pool5',
                                  top='loss3_drop',
                                  dropout_param=drop_param_0_4)
    layers += [loss3_drop]

    loss3_fcrnn = IPRNN_v2('loss3_fcrnn', 'loss3_drop',
                        'loss3_fcrnn',
                        [get_ipp(1024), get_ipp(1024)], ep_0_5, params,
                        BlobShape(batch_size, 1024))
    loss3_fcrnn.ip.set('name', 'loss3/classifier1')
    layers += loss3_fcrnn.__layers__

    loss3_class = LayerParameter('InnerProduct',
                                   name='loss3/classifier101',
                                   bottom='loss3_fcrnn',
                                   top='loss3_class',
                                   inner_product_param=get_ipp(101),
                                   param=params[0])
    loss3_loss = LayerParameter('SoftmaxWithLoss',
                                  name='loss3_loss',
                                  bottom=['loss3_class', 'label'],
                                  top='loss3_loss')
    loss3_top1 = LayerParameter('Accuracy',
                                  name='loss3_top1',
                                  bottom=['loss3_class', 'label'],
                                  top='loss3_top1',
                                  include=include_test)
    loss3_top5 = LayerParameter('Accuracy',
                                  name='loss3_top5',
                                  bottom=['loss3_class', 'label'],
                                  top='loss3_top5',
                                  include=include_test,
                                  accuracy_param=acc_param_5)
    layers += [loss3_class,
               loss3_loss, loss3_top1,
               loss3_top5]
    print display(layers)
