'''
@Author: xxxmy
@Github: github.com/VectXmy
@Date: 2019-09-26
@Email: xxxmy@foxmail.com
'''

import cv2
from model.fcos import FCOSDetector
import torch
from torchvision import transforms
import numpy as np
from dataloader.VOC_dataset import VOCDataset
from dataloader.COCO_dataset import COCODataset
from dataloader.NW import NWDataset
from dataloader.DIOR import DIORDataset
import time
import numpy as np
# color=np.random.randint(0,255,[20,3])

def preprocess_img(image,input_ksize):
    '''
    resize image and bboxes 
    Returns
    image_paded: input_ksize  
    bboxes: [None,4]
    '''
    min_side, max_side    = input_ksize
    h,  w, _  = image.shape

    smallest_side = min(w,h)
    largest_side=max(w,h)
    scale=min_side/smallest_side
    if largest_side*scale>max_side:
        scale=max_side/largest_side
    nw, nh  = int(scale * w), int(scale * h)
    image_resized = cv2.resize(image, (nw, nh))

    pad_w=32-nw%32
    pad_h=32-nh%32

    image_paded = np.zeros(shape=[nh+pad_h, nw+pad_w, 3],dtype=np.float32)
    image_paded[:nh, :nw, :] = image_resized
    return image_paded
    
def convertSyncBNtoBN(module):
    module_output = module
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module_output = torch.nn.BatchNorm2d(module.num_features,
                                               module.eps, module.momentum,
                                               module.affine,
                                               module.track_running_stats)
        if module.affine:
            module_output.weight.data = module.weight.data.clone().detach()
            module_output.bias.data = module.bias.data.clone().detach()
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
    for name, child in module.named_children():
        module_output.add_module(name,convertSyncBNtoBN(child))
    del module
    return module_output
if __name__=="__main__":
    import random
    col=[]
    # for _ in range(20):
    #     col.append((random.randint(0,255),random.randint(0,255),random.randint(0,255)))
    # print(col)
    col=[(43, 170, 255), (22, 179, 255), (255, 176, 71), (59, 145, 159), (138, 208, 48), (129, 200, 203),
     (218, 85, 38), (2, 209, 128), (115, 168, 80), (202, 196, 53), (142, 253, 38), (167, 145, 137), (
        142, 89, 70), (196, 255, 112), (234, 128, 211), (70, 33, 241), (7, 160, 95), (137, 203, 238), 
     (48, 231, 116), (166, 138, 150)]
    # class DefaultConfig():
    # #backbone
    #     pretrained=True
    #     freeze_stage_1=False
    #     freeze_bn=False

    #     #fpn
    #     fpn_out_channels=256
    #     use_p5=True
        
    #     #head
    #     class_num=10
    #     use_GN_head=True
    #     prior=0.01
    #     add_centerness=False
    #     cnt_on_reg=False

    #     #training
    #     strides=[8,16,32,64,128]
    #     # limit_range=[[-1,64],[64,128],[128,256],[256,512],[512,999999]]
    #     limit_range=[[-1,64],[64,192],[128,192],[192,256],[256,999999]]
    #     #inference
    #     score_threshold=0.3
    #     nms_iou_threshold=0.5
    #     max_detection_boxes_num=150
        
    #     ##scene_head_para
    #     input_channel=512
    #     output_channel=512
    #     pooling_size=3
    #     output_class=10
    # from model.config import DefaultConfig

    # DefaultConfig.score_threshold=0.3

    model=FCOSDetector(mode="inference").cuda()
    model=torch.nn.DataParallel(model)
    # model=torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    # print("INFO===>success convert BN to SyncBN")
    model.load_state_dict(torch.load("shidudi1e4Diormixupmuti_label_simgoid0.4_0.0001_50_C3_C4_C5_conc_max_pooling_192fcos_8001024_epoch60_loss0.0774.pth",map_location=torch.device('cuda')),False)
    
    # model=convertSyncBNtoBN(model)
    # print("INFO===>success convert SyncBN to BN")
    model=model.cuda().eval()
    print("===>success loading model")

    import os
    root="./test_images/"
    names=os.listdir(root)
    for name in names:
        img_bgr=cv2.imread(root+name)
        img_pad=preprocess_img(img_bgr,[800,1024])
        # img=cv2.cvtColor(img_pad.copy(),cv2.COLOR_BGR2RGB)
        img=img_pad.copy()
        img_t=torch.from_numpy(img).float().permute(2,0,1)
        img1= transforms.Normalize([102.9801, 115.9465, 122.7717],[1.,1.,1.])(img_t)
        # img1=transforms.ToTensor()(img1)
        # img1= transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225),inplace=True)(img1)
        img1=img1.cuda()
        

        start_t=time.time()
        with torch.no_grad():
            out=model(img1.unsqueeze_(dim=0))
        end_t=time.time()
        cost_t=1000*(end_t-start_t)
        print("===>success processing img, cost time %.2f ms"%cost_t)
        # print(out)
        scores,classes,boxes,s=out

        boxes=boxes[0].cpu().numpy().tolist()
        classes=classes[0].cpu().numpy().tolist()
        scores=scores[0].cpu().numpy().tolist()

        for i,box in enumerate(boxes):
            pt1=(int(box[0]),int(box[1]))
            pt2=(int(box[2]),int(box[3]))
            # col=(color[int(classes[i])-1][0],color[int(classes[i])-1][1],color[int(classes[i])-1][2])
            # print(col)
            img_pad=cv2.rectangle(img_pad,pt1,pt2,col[int(classes[i])-1],3)
            # img_pad=cv2.putText(img_pad,"%s %.3f"%(DIORDataset.CLASSES_NAME[int(classes[i])],scores[i]),(int(box[0]),int(box[1])+10),cv2.FONT_HERSHEY_SIMPLEX,0.5,col[int(classes[i])-1],2)
        cv2.imwrite("./out_images2/"+name,img_pad[:800,:800])





