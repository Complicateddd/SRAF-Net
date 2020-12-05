#import torch
#def _post_process(preds_topk):
#        '''
#        cls_scores_topk [batch_size,max_num]
#        cls_classes_topk [batch_size,max_num]
#        boxes_topk [batch_size,max_num,4]
#        '''
#        _cls_scores_post=[]
#        _cls_classes_post=[]
#        _boxes_post=[]
#        cls_scores_topk,cls_classes_topk,boxes_topk=preds_topk
#        for batch in range(cls_classes_topk.shape[0]):
#            mask=cls_scores_topk[batch]>=0.01
#            _cls_scores_b=cls_scores_topk[batch][mask]#[?]
#            _cls_classes_b=cls_classes_topk[batch][mask]#[?]
#            _boxes_b=boxes_topk[batch][mask]#[?,4]
#            
#            nms_ind=box_nms(_boxes_b,_cls_scores_b,0.01)
#            _cls_scores_post.append(_cls_scores_b[nms_ind])
#            _cls_classes_post.append(_cls_classes_b[nms_ind])
#            _boxes_post.append(_boxes_b[nms_ind])
#        scores,classes,boxes= torch.stack(_cls_scores_post,dim=0),torch.stack(_cls_classes_post,dim=0),torch.stack(_boxes_post,dim=0)
#def box_nms(boxes,scores,thr):
#        '''
#        boxes: [?,4]
#        scores: [?]
#        '''
#        if boxes.shape[0]==0:
#            return torch.zeros(0,device=boxes.device).long()
#        assert boxes.shape[-1]==4
#        x1,y1,x2,y2=boxes[:,0],boxes[:,1],boxes[:,2],boxes[:,3]
#        areas=(x2-x1+1)*(y2-y1+1)
#        order=scores.sort(0,descending=True)[1]
#        
#        keep=[]
#        while order.numel()>0:
#            print("order:{}".format(order))
#            print("order.numel:{}".format(order.numel()))
#            if order.numel()==1:
#                i=order.item()
#                keep.append(i)
#                break
#            else:
#                i=order[0].item()
#                keep.append(i)
#            
#            xmin=x1[order[1:]].clamp(min=float(x1[i]))
#            ymin=y1[order[1:]].clamp(min=float(y1[i]))
#            xmax=x2[order[1:]].clamp(max=float(x2[i]))
#            ymax=y2[order[1:]].clamp(max=float(y2[i]))
#            inter=(xmax-xmin).clamp(min=0)*(ymax-ymin).clamp(min=0)
#            iou=inter/(areas[i]+areas[order[1:]]-inter)
#            print("iou",iou)
#            idx=(iou<=thr).nonzero().squeeze()
#            print("idx",idx)
#            if idx.numel()==0:
#                break
#            order=order[idx+1]
#        return torch.LongTensor(keep)
#if __name__ == '__main__':
#    # cls_scores_topk [batch_size,max_num]
#    #     cls_classes_topk [batch_size,max_num]
#    #     boxes_topk [batch_size,max_num,4]
#    # a=torch.tensor([[1,2,3,4],[1.1,2.2,3.3,4.4],[3,4,5,6],[7,8,9,10]])
#    # score=torch.tensor([0.5,0.7,0.1,0.6])
#    # a=torch.Tensor(4,35)
#    # # ind=torch.topk(a,a.shape[-1],dim=-1,largest=True,sorted=True)[1]
#    # # print(a.sort(0,descending=True)[1])
#    # # print(ind[0].item())
#    # # longt=box_nms(a,score,0.1)
#    # # print(longt)
#    # b=a[0]
#    # b1=a[1][:3]
#    # c=[]
#    # c.append(b)
#    # c.append(b1)
#    # print(torch.stack(c,dim=0).shape)
#    cls_scores_topk=torch.tensor([1,3,5,2,18,15])
#    # cls_classes_topk=torch.LongTensor(3,30)
#    # boxes_topk=torch.Tensor(3,30,4)
#    # _post_process([cls_scores_topk,cls_classes_topk,boxes_topk])
#    g=[(-cls_scores_topk).argsort() for index, score in enumerate(cls_scores_topk)]
#    for sample_boxes, mask in zip(cls_scores_topk, g):
#        print(sample_boxes,mask)
#    # pred_scores = [sample_boxes[mask] for sample_boxes, mask in zip(cls_scores_topk, g)]
#    print(g)
## print(a.sigmoid())

import torch
def bbox_overlaps_diou(bboxes1, bboxes2):

    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    dious = torch.zeros((rows, cols))
    if rows * cols == 0:
        return dious
    exchange = False
    if bboxes1.shape[0] > bboxes2.shape[0]:
        bboxes1, bboxes2 = bboxes2, bboxes1
        dious = torch.zeros((cols, rows))
        exchange = True

    w1 = bboxes1[:, 2] - bboxes1[:, 0]
    h1 = bboxes1[:, 3] - bboxes1[:, 1]
    w2 = bboxes2[:, 2] - bboxes2[:, 0]
    h2 = bboxes2[:, 3] - bboxes2[:, 1]

    area1 = w1 * h1
    area2 = w2 * h2
    center_x1 = (bboxes1[:, 2] + bboxes1[:, 0]) / 2
    center_y1 = (bboxes1[:, 3] + bboxes1[:, 1]) / 2
    center_x2 = (bboxes2[:, 2] + bboxes2[:, 0]) / 2
    center_y2 = (bboxes2[:, 3] + bboxes2[:, 1]) / 2

    inter_max_xy = torch.min(bboxes1[:, 2:],bboxes2[:, 2:])
    inter_min_xy = torch.max(bboxes1[:, :2],bboxes2[:, :2])
    out_max_xy = torch.max(bboxes1[:, 2:],bboxes2[:, 2:])
    out_min_xy = torch.min(bboxes1[:, :2],bboxes2[:, :2])

    inter = torch.clamp((inter_max_xy - inter_min_xy), min=0)
    inter_area = inter[:, 0] * inter[:, 1]
    inter_diag = (center_x2 - center_x1)**2 + (center_y2 - center_y1)**2
    outer = torch.clamp((out_max_xy - out_min_xy), min=0)
    outer_diag = (outer[:, 0] ** 2) + (outer[:, 1] ** 2)
    union = area1+area2-inter_area
    dious = inter_area / union - (inter_diag) / outer_diag
    dious = torch.clamp(dious,min=-1.0,max = 1.0)
    if exchange:
        dious = dious.T
    return dious

def iou_loss(preds,targets):
    '''
    Args:
    preds: [n,4] ltrb
    targets: [n,4]
    '''
    lt=torch.min(preds[:,:2],targets[:,:2])
    rb=torch.min(preds[:,2:],targets[:,2:])
    wh=(rb+lt).clamp(min=0)
    overlap=wh[:,0]*wh[:,1]#[n]
    area1=(preds[:,2]+preds[:,0])*(preds[:,3]+preds[:,1])
    area2=(targets[:,2]+targets[:,0])*(targets[:,3]+targets[:,1])
    iou=overlap/(area1+area2-overlap)
    loss=-iou.clamp(min=1e-6).log()
    return loss.sum()

def giou_loss(preds,targets):
    '''
    Args:
    preds: [n,4] ltrb
    targets: [n,4]
    '''
    lt_min=torch.min(preds[:,:2],targets[:,:2])
    rb_min=torch.min(preds[:,2:],targets[:,2:])
    wh_min=(rb_min+lt_min).clamp(min=0)
    overlap=wh_min[:,0]*wh_min[:,1]#[n]
    area1=(preds[:,2]+preds[:,0])*(preds[:,3]+preds[:,1])
    area2=(targets[:,2]+targets[:,0])*(targets[:,3]+targets[:,1])
    union=(area1+area2-overlap)
    iou=overlap/union

    lt_max=torch.max(preds[:,:2],targets[:,:2])
    rb_max=torch.max(preds[:,2:],targets[:,2:])
    wh_max=(rb_max+lt_max).clamp(0)
    G_area=wh_max[:,0]*wh_max[:,1]#[n]

    giou=iou-(G_area-union)/G_area.clamp(1e-10)
    loss=1.-giou
    return loss.sum()
















