class DefaultConfig():
    #backbone
    pretrained=True
    freeze_stage_1=False
    freeze_bn=False

    #fpn
    fpn_out_channels=256
    use_p5=True
    
    #head
    class_num=20
    use_GN_head=True
    prior=0.01
    add_centerness=False
    cnt_on_reg=False

    #training
    strides=[8,16,32,64,128]
    limit_range=[[-1,64],[64,128],[128,192],[192,256],[256,999999]]
    
    #inference
    score_threshold=0.4
    nms_iou_threshold=0.5
    max_detection_boxes_num=150
    
    ##scene_head_para
    input_channel=512
    output_channel=512
    pooling_size=3
    output_class=20
    scene_use_GN=False
    muti_lr=False

    # data augment
    mixup=True