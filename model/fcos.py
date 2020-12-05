'''
@Author: xxxmy
@Github: github.com/VectXmy
@Date: 2019-09-26
@Email: xxxmy@foxmail.com
'''

from .head import ClsCntRegHead
from .fpn import FPN
from .SceneEnhanced_FPN import SE_FPN
from .backbone.resnet import resnet50,resnet18,resnet101
import torch.nn as nn
from .loss import GenTargets,LOSS,coords_fmap2orig,focal_loss_with_scene
import torch
from .config import DefaultConfig
from model.TextureSceneMudule import TextureSceneMudule
import torch.nn.functional as F
import numpy as np
import torch_extension
_nms = torch_extension.nms


class SEAF(nn.Module):
    
    def __init__(self,config=None):
        super().__init__()
        if config is None:
            config=DefaultConfig
        self.backbone=resnet101(pretrained=config.pretrained,if_include_top=False)
        # self.fpn=FPN(config.fpn_out_channels,use_p5=config.use_p5)
        self.head=ClsCntRegHead(config.fpn_out_channels,config.class_num,
                                config.use_GN_head,config.cnt_on_reg,config.prior)
        # self.scene_head=TextureSceneMudule(config.input_channel,config.output_channel,
        #     config.pooling_size,config.output_class,config.scene_use_GN)
        self.se_fpn=SE_FPN(config)
        self.config=config
    def train(self,mode=True):
        '''
        set module training mode, and frozen bn
        '''
        super().train(mode=True)
        def freeze_bn(module):
            if isinstance(module,nn.BatchNorm2d):
                module.eval()
            classname = module.__class__.__name__
            if classname.find('BatchNorm') != -1:
                for p in module.parameters(): p.requires_grad=False
        if self.config.freeze_bn:
            self.apply(freeze_bn)
            print("INFO===>success frozen BN")
        if self.config.freeze_stage_1:
            self.backbone.freeze_stages(1)
            print("INFO===>success frozen backbone stage1")

    def forward(self,x):
        '''
        Returns
        list [cls_logits,cnt_logits,reg_preds]  
        cls_logits  list contains five [batch_size,class_num,h,w]
        cnt_logits  list contains five [batch_size,1,h,w]
        reg_preds   list contains five [batch_size,4,h,w]
        '''
        C3,C4,C5=self.backbone(x)
        P_list,SL_pred=self.se_fpn(C3,C4,C5)
        cls_logits,cnt_logits,reg_preds=self.head(P_list)
        return [cls_logits,cnt_logits,reg_preds],SL_pred
        

class DetectHead(nn.Module):
    def __init__(self,score_threshold,nms_iou_threshold,max_detection_boxes_num,strides,config=None):
        super().__init__()
        self.score_threshold=score_threshold
        self.nms_iou_threshold=nms_iou_threshold
        self.max_detection_boxes_num=max_detection_boxes_num
        self.strides=strides
        if config is None:
            self.config=DefaultConfig
        else:
            self.config=config
    def forward(self,inputs,s_pred):
        '''
        inputs  list [cls_logits,cnt_logits,reg_preds]  
        cls_logits  list contains five [batch_size,class_num,h,w]  
        cnt_logits  list contains five [batch_size,1,h,w]  
        reg_preds   list contains five [batch_size,4,h,w] 
        '''
        cls_logits,coords=self._reshape_cat_out(inputs[0],self.strides)#[batch_size,sum(_h*_w),class_num]
        
        cnt_logits,_=self._reshape_cat_out(inputs[1],self.strides)#[batch_size,sum(_h*_w),1]
        reg_preds,_=self._reshape_cat_out(inputs[2],self.strides)#[batch_size,sum(_h*_w),4]
        x_s=s_pred
        s_sigmod=torch.sigmoid(x_s)
        a = torch.zeros_like(s_sigmod)
        b = torch.ones_like(s_sigmod)

        x=torch.where(s_sigmod>0.001,b,a)
        # print(x)
        # mask_p=(s_sigmod>=0.4).reshape(-1)
        # mask_n=(s_sigmod<0.1).reshape(-1)
        # cls_predss=cls_logits.sigmoid_()
        # cls_preds=

        # cls_preds=torch.where(x_s==0,predss1*0.8,predss1)


        cls_predss=cls_logits.sigmoid_()
        cls_preds=0.95*cls_predss+0.05*x
        # cls_preds[:,:,mask_p]+=0.
        # cls_preds[cls_preds>1]=1.
        # cls_preds[:,:,mask_n]-=0.02
        # cls_preds[cls_preds<0]=0.

        cnt_preds=cnt_logits.sigmoid_()
        # print(cls_preds.shape)


        cls_scores,cls_classes=torch.max(cls_preds,dim=-1)#[batch_size,sum(_h*_w)]
        if self.config.add_centerness:
            cls_scores=cls_scores*(cnt_preds.squeeze(dim=-1))#[batch_size,sum(_h*_w)]
        cls_classes=cls_classes+1#[batch_size,sum(_h*_w)]

        boxes=self._coords2boxes(coords,reg_preds)#[batch_size,sum(_h*_w),4]

        #select topk
        max_num=min(self.max_detection_boxes_num,cls_scores.shape[-1])
        topk_ind=torch.topk(cls_scores,max_num,dim=-1,largest=True,sorted=True)[1]#[batch_size,max_num]
        _cls_scores=[]
        _cls_classes=[]
        _boxes=[]
        for batch in range(cls_scores.shape[0]):
            _cls_scores.append(cls_scores[batch][topk_ind[batch]])#[max_num]
            _cls_classes.append(cls_classes[batch][topk_ind[batch]])#[max_num]
            _boxes.append(boxes[batch][topk_ind[batch]])#[max_num,4]
        cls_scores_topk=torch.stack(_cls_scores,dim=0)#[batch_size,max_num]
        cls_classes_topk=torch.stack(_cls_classes,dim=0)#[batch_size,max_num]
        boxes_topk=torch.stack(_boxes,dim=0)#[batch_size,max_num,4]
        assert boxes_topk.shape[-1]==4
        return self._post_process([cls_scores_topk,cls_classes_topk,boxes_topk])

    def _post_process(self,preds_topk):
        '''
        cls_scores_topk [batch_size,max_num]
        cls_classes_topk [batch_size,max_num]
        boxes_topk [batch_size,max_num,4]
        '''
        _cls_scores_post=[]
        _cls_classes_post=[]
        _boxes_post=[]
        cls_scores_topk,cls_classes_topk,boxes_topk=preds_topk
        for batch in range(cls_classes_topk.shape[0]):
            mask=cls_scores_topk[batch]>=self.score_threshold
            _cls_scores_b=cls_scores_topk[batch][mask]#[?]
            _cls_classes_b=cls_classes_topk[batch][mask]#[?]
            _boxes_b=boxes_topk[batch][mask]#[?,4]
            # nms_ind=self.batched_nms(_boxes_b,_cls_scores_b,_cls_classes_b,self.nms_iou_threshold)
            # nms_ind=self.soft_nms_pytorch(_boxes_b,_cls_scores_b)
            nms_ind=self.box_nms(_boxes_b,_cls_scores_b,self.nms_iou_threshold)
            _cls_scores_post.append(_cls_scores_b[nms_ind])
            _cls_classes_post.append(_cls_classes_b[nms_ind])
            _boxes_post.append(_boxes_b[nms_ind])
        scores,classes,boxes= torch.stack(_cls_scores_post,dim=0),torch.stack(_cls_classes_post,dim=0),torch.stack(_boxes_post,dim=0)
        
        return scores,classes,boxes
    # @staticmethod
    # def box_nms(boxes, scores, nms_thresh, max_count=-1):
    #     '''Performs non-maximum suppression, run on GPU or CPU according to
    #     boxes's device.
    #     Args:
    #         boxes(Tensor): `xyxy` mode boxes, use absolute coordinates(or relative coordinates), shape is (n, 4)
    #         scores(Tensor): scores, shape is (n, )
    #         nms_thresh(float): thresh
    #         max_count (int): if > 0, then only the top max_proposals are kept  after non-maximum suppression
    #     Returns:
    #         indices kept.'''
    
    #     keep = _nms(boxes, scores, nms_thresh)
    #     if max_count > 0:
    #         keep = keep[:max_count]
    #     return keep
    @staticmethod
    def box_nms(boxes,scores,thr):
        '''
        boxes: [?,4]
        scores: [?]
        '''
        if boxes.shape[0]==0:
            return torch.zeros(0,device=boxes.device).long()
        assert boxes.shape[-1]==4
        x1,y1,x2,y2=boxes[:,0],boxes[:,1],boxes[:,2],boxes[:,3]
        areas=(x2-x1+1)*(y2-y1+1)
        order=scores.sort(0,descending=True)[1]
        keep=[]
        while order.numel()>0:
            if order.numel()==1:
                i=order.item()
                keep.append(i)
                break
            else:
                i=order[0].item()
                keep.append(i)
            
            xmin=x1[order[1:]].clamp(min=float(x1[i]))
            ymin=y1[order[1:]].clamp(min=float(y1[i]))
            xmax=x2[order[1:]].clamp(max=float(x2[i]))
            ymax=y2[order[1:]].clamp(max=float(y2[i]))
            inter=(xmax-xmin).clamp(min=0)*(ymax-ymin).clamp(min=0)
            iou=inter/(areas[i]+areas[order[1:]]-inter)
            idx=(iou<=thr).nonzero().squeeze()
            if idx.numel()==0:
                break
            order=order[idx+1]
        return torch.LongTensor(keep)

    @staticmethod
    def batched_nms(self,boxes,scores,idxs,iou_threshold=0.5):
        
        if boxes.numel() == 0:
            return torch.empty((0,), dtype=torch.int64, device=boxes.device)
        # strategy: in order to perform NMS independently per class.
        # we add an offset to all the boxes. The offset is dependent
        # only on the class idx, and is large enough so that boxes
        # from different classes do not overlap
        max_coordinate = boxes.max()
        offsets = idxs.to(boxes) * (max_coordinate + 1)
        boxes_for_nms = boxes + offsets[:, None]
        keep = self.box_nms(boxes_for_nms, scores, iou_threshold)
        return keep
    
    @staticmethod
    def soft_nms_pytorch(dets=None, box_scores=None, sigma=0.2, thresh=0.001, cuda=1):
    
    # Build a pytorch implement of Soft NMS algorithm.
    # # Augments
    #     dets:        boxes coordinate tensor (format:[y1, x1, y2, x2])
    #     box_scores:  box score tensors
    #     sigma:       variance of Gaussian function
    #     thresh:      score thresh
    #     cuda:        CUDA flag
    # # Return
    #     the index of the selected boxes
    # Indexes concatenate boxes with the last column
        # print(dets)
        N = dets.shape[0]
        if cuda:
            indexes = torch.arange(0, N, dtype=torch.float).cuda().view(N, 1)
        else:
            indexes = torch.arange(0, N, dtype=torch.float).view(N, 1)
        dets = torch.cat((dets, indexes), dim=1)

        # The order of boxes coordinate is [y1,x1,y2,x2]
        y1 = dets[:, 0]
        x1 = dets[:, 1]
        y2 = dets[:, 2]
        x2 = dets[:, 3]
        scores = box_scores
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)

        for i in range(N):
            # intermediate parameters for later parameters exchange
            tscore = scores[i].clone()
            pos = i + 1

            if i != N - 1:
                maxscore, maxpos = torch.max(scores[pos:], dim=0)
                if tscore < maxscore:
                    dets[i], dets[maxpos.item() + i + 1] = dets[maxpos.item() + i + 1].clone(), dets[i].clone()
                    scores[i], scores[maxpos.item() + i + 1] = scores[maxpos.item() + i + 1].clone(), scores[i].clone()
                    areas[i], areas[maxpos + i + 1] = areas[maxpos + i + 1].clone(), areas[i].clone()

            # IoU calculate
            yy1 = np.maximum(dets[i, 0].to("cpu").numpy(), dets[pos:, 0].to("cpu").numpy())
            xx1 = np.maximum(dets[i, 1].to("cpu").numpy(), dets[pos:, 1].to("cpu").numpy())
            yy2 = np.minimum(dets[i, 2].to("cpu").numpy(), dets[pos:, 2].to("cpu").numpy())
            xx2 = np.minimum(dets[i, 3].to("cpu").numpy(), dets[pos:, 3].to("cpu").numpy())
            
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = torch.tensor(w * h).cuda() if cuda else torch.tensor(w * h)
            ovr = torch.div(inter, (areas[i] + areas[pos:] - inter))

            # Gaussian decay
            weight = torch.exp(-(ovr * ovr) / sigma)
            scores[pos:] = weight * scores[pos:]

        # select the boxes and keep the corresponding indexes
        keep = dets[:, 4][scores > thresh].long().cpu()

        return torch.LongTensor(keep)


    def _coords2boxes(self,coords,offsets):
        '''
        Args
        coords [sum(_h*_w),2]
        offsets [batch_size,sum(_h*_w),4] ltrb
        '''
        x1y1=coords[None,:,:]-offsets[...,:2]
        x2y2=coords[None,:,:]+offsets[...,2:]#[batch_size,sum(_h*_w),2]
        boxes=torch.cat([x1y1,x2y2],dim=-1)#[batch_size,sum(_h*_w),4]
        return boxes


    def _reshape_cat_out(self,inputs,strides):
        '''
        Args
        inputs: list contains five [batch_size,c,_h,_w]
        Returns
        out [batch_size,sum(_h*_w),c]
        coords [sum(_h*_w),2]
        '''
        batch_size=inputs[0].shape[0]
        c=inputs[0].shape[1]
        out=[]
        coords=[]
        for pred,stride in zip(inputs,strides):
            pred=pred.permute(0,2,3,1)
            coord=coords_fmap2orig(pred,stride).to(device=pred.device)
            pred=torch.reshape(pred,[batch_size,-1,c])
            out.append(pred)
            coords.append(coord)
        return torch.cat(out,dim=1),torch.cat(coords,dim=0)

class ClipBoxes(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,batch_imgs,batch_boxes):
        batch_boxes=batch_boxes.clamp_(min=0)
        h,w=batch_imgs.shape[2:]
        batch_boxes[...,[0,2]]=batch_boxes[...,[0,2]].clamp_(max=w-1)
        batch_boxes[...,[1,3]]=batch_boxes[...,[1,3]].clamp_(max=h-1)
        return batch_boxes

        
class FCOSDetector(nn.Module):
    def __init__(self,mode="training",config=None):
        super().__init__()
        if config is None:
            self.config=DefaultConfig
        self.mode=mode
        self.fcos_body=SEAF(config=self.config)
        # self.add_loss=nn.CrossEntropyLoss()
        self.add_loss=nn.BCELoss()
        if mode=="training":
            self.target_layer=GenTargets(strides=self.config.strides,limit_range=self.config.limit_range)
            self.loss_layer=LOSS()
#            self.add_loss=focal_loss_with_scene()
        elif mode=="inference":
            self.detection_head=DetectHead(self.config.score_threshold,self.config.nms_iou_threshold,
                                            self.config.max_detection_boxes_num,self.config.strides,config)
            self.clip_boxes=ClipBoxes()
        
    
    def forward(self,inputs):
        '''
        inputs 
        [training] list  batch_imgs,batch_boxes,batch_classes,batch_scene_classes
        [inference] img
        '''

        if self.mode=="training":
            
            batch_imgs,batch_boxes,batch_classes,batch_scene_classes=inputs
            out,s_pred=self.fcos_body(batch_imgs)
            # batch_scene_classes=batch_scene_classes*0.95+0.05/10
            # batch_scene_classes=batch_scene_classes.squeeze(-1)
            
            # batch_scene_classes=(torch.arange(1,self.config.output_class+1,device=batch_scene_classes.device)[None,:]==batch_scene_classes).float()
            # s_pred=F.softmax(s_pred,dim=1)
            # print(s_pred)
            # print(batch_scene_classes)
            # scene_loss=focal_loss_with_scene(s_pred,batch_scene_classes).mean()
            scene_loss=self.add_loss(torch.sigmoid(s_pred),batch_scene_classes)
            
            ###add_最后的信息加入：
            # x_s=torch.sigmoid(s_pred)
            # mask_p=(x_s>=0.4)
            # # print(mask_p.shape)
            # mask_n=(x_s<0.4)

            # for i in out[0]:
            #     i=i.sigmoid()
                # i[mask_p]+=0.1
                # mask1=i>1.0
                # i[mask1]=0.95
                # i[mask_n]-=0.1
                # mask2=i<0.0
                # i[mask2]=0.1
            #################################################

            targets=self.target_layer([out,batch_boxes,batch_classes])
            losses=self.loss_layer([out,targets,s_pred])
            return losses,scene_loss
        
        elif self.mode=="inference":
            # raise NotImplementedError("no implement inference model")
            '''
            for inference mode, img should preprocessed before feeding in net 
            '''
            batch_imgs=inputs
            out,s_pred=self.fcos_body(batch_imgs)
            # print(s_pred.shape)
            scores,classes,boxes=self.detection_head(out,s_pred)
            boxes=self.clip_boxes(batch_imgs,boxes)
            return scores,classes,boxes,F.sigmoid(s_pred)



    


