from grasp_methods.affga_net.sgdn.models.affga.backbone import resnet, xception, drn, mobilenet


def build_backbone(input_channels, backbone, output_stride, BatchNorm):

    if backbone == 'resnet':
        return resnet.ResNet101(input_channels, output_stride, BatchNorm)

    elif backbone == 'xception':
        return xception.AlignedXception(output_stride, BatchNorm)
    elif backbone == 'drn':
        return drn.drn_d_54(BatchNorm)
    elif backbone == 'mobilenet':
        return mobilenet.MobileNetV2(output_stride, BatchNorm)
    else:
        raise NotImplementedError
