# import torch
# from torch.autograd import Variable
# import numpy as np
# import matplotlib.pyplot as plt
# def coords_fmap2orig(feature,stride):
#     '''
#     # transfor one fmap coords to orig coords
#     # Args
#     # featurn [batch_size,h,w,c]
#     # stride int
#     # Returns 
#     # coords [n,2]
#     '''
#     h,w=feature.shape[1:3]
#     shifts_x = torch.arange(0, w * stride, stride, dtype=torch.float32)
#     shifts_y = torch.arange(0, h * stride, stride, dtype=torch.float32)

#     shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
#     shift_x = torch.reshape(shift_x, [-1])-0.5
#     shift_y = torch.reshape(shift_y, [-1])-0.5
#     coords = torch.stack([shift_x, shift_y], -1) + stride // 2
#     return coords

# feature=torch.Tensor(1,3,3,3)
# print(feature.shape)
# coords=coords_fmap2orig(feature,2)
# print(coords)
# # # temp=torch.rand(2,5)
# # # a=torch.tensor([[1,3,4,3,2],[1,3,4,3,2]])
# # # b=torch.zeros((3,5),dtype=torch.long).scatter_(0,a,temp)
# # # # l=[]
# # # # l.append(a)
# # # # l.append(a)

# # # # print(l)
# # # score_seq = [(-score).argsort() for index, score in enumerate(l)]
# # # # for index, score in enumerate(a):
# # # # 	print(index,score)
# # # # pred_boxes = [sample_boxes[mask] for sample_boxes, mask in zip(pred_boxes, score_seq)]
# # # # pred_labels = [sample_boxes[mask] for sample_boxes, mask in zip(pred_labels, score_seq)]
# # # pred_scores = [sample_boxes[mask] for sample_boxes, mask in zip(l, score_seq)]
# # # print(a)
# # # print(b)
# # # x = torch.Tensor([16,55])
# # # x=Variable(x,requires_grad=True)
# # # y=x*x
# # # y.mean()
# # # print(y)
# # # # y.backward()
# # # print(y.mean())
# # # target_pos=5
# # # target_pos=(torch.arange(1,10+1)[None,:]==target_pos).float()
# # # print(target_pos)
# # #tensor([[0.1940, 0.3340, 0.8184, 0.4269, 0.5945],
# # #        [0.2078, 0.5978, 0.0074, 0.0943, 0.0266]])
# # # b=torch.tensor([[0, 1, 2, 0, 0], [2, 0, 0, 1, 2]])

											  
# # # b=torch.zeros_like(x,dtype=torch.long).scatter_(-1, b.unsqueeze(dim=-1), 1)
# # # b1=b>0
# # # print(x)
# # # print(b)
# # # print(b1)
# # # print(b.shape)
# # # for sample_boxes, mask in zip(l, score_seq):
# # # 	print(sample_boxes,mask)
# # def sort_by_score(pred_boxes, pred_labels, pred_scores):
# #     score_seq = [(-score).argsort() for index, score in enumerate(pred_scores)]
# #     pred_boxes = [sample_boxes[mask] for sample_boxes, mask in zip(pred_boxes, score_seq)]
# #     pred_labels = [sample_boxes[mask] for sample_boxes, mask in zip(pred_labels, score_seq)]
# #     pred_scores = [sample_boxes[mask] for sample_boxes, mask in zip(pred_scores, score_seq)]
# #     return pred_boxes, pred_labels, pred_scores
# # # a=np.zeros((0,))
# # # print(a)
# # # a=np.append(a,1)
# # # a=np.append(a,1)

# # def _compute_ap(recall, precision):
# #     """ Compute the average precision, given the recall and precision curves.
# #     Code originally from https://github.com/rbgirshick/py-faster-rcnn.
# #     # Arguments
# #         recall:    The recall curve (list).
# #         precision: The precision curve (list).
# #     # Returns
# #         The average precision as computed in py-faster-rcnn.
# #     """
# #     # correct AP calculation
# #     # first append sentinel values at the end
# #     mrec = np.concatenate(([0.], recall, [1.]))
# #     print(mrec)
# #     print('\n')
# #     mpre = np.concatenate(([0.], precision, [0.]))
# #     print(mpre)
# #     print('\n')
# #     # compute the precision envelope
# #     for i in range(mpre.size - 1, 0, -1):
# #         mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
# #         # print(mpre[i-1])
# #     print(mpre)
# #     print('\n')
# #     # to calculate area under PR curve, look for points
# #     # where X axis (recall) changes value
# #     i = np.where(mrec[1:] != mrec[:-1])[0]
# #     print(mrec[1:])
# #     print('\n')
# #     print( mrec[:-1])
# #     print('\n')
# #     print(i)
# #     plt.figure()
# #     plt.plot(mrec,mpre)
# #     plt.show()
# #     # and sum (\Delta recall) * prec
# #     ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
# #     return ap

# # tp=np.array([1,1,2,3,3])
# # fp=np.array([0,1,1,2,3])
# # tp=tp/4
# # # print("recall:{}".format(tp))
# # fp=tp/(tp+fp)
# # # print(fp)
# # # print("precision:{}".format(fp.size))
# # _compute_ap(tp,fp)
# # # for i in range(10, 0, -1):
# # # 	print(i)
# # # pred_boxes=[]
# # # pred_labels=[]
# # # pred_scores=[]
# # # a=torch.tensor([[213,324,344,633],[123,324,344,633],[567,324,344,633]])
# # # label=torch.tensor([2,2,3])
# # # score=torch.tensor([0.8,0.5,0.9])
# # # for _ in range(2):
# # # 	pred_boxes.append(a)
# # # 	pred_labels.append(label)
# # # 	pred_scores.append(score)
# # # # print(pred_scores)
# # # # print(pred_boxes)

# # # out1,out2,out3=sort_by_score(pred_boxes,pred_labels,pred_scores)
# # # print(out2)
# # # p=[simple==2 for simple in out2]
# # # print(p)
# # # gt_single_cls = [sample_boxes[mask] for sample_boxes, mask in zip(out1, p)]
# # # print(gt_single_cls)
# # a=np.array([0.,0.125 ,0.125, 0.25 , 0.375, 0.375 ,0.5 ,  0.5 ,  0.5 ,  0.625 ,1.   ])


# # b=np.array([1.     ,    1.    ,     0.2    ,    0.2  ,      0.15789474, 0.11111111,
# #  0.11111111 ,0.09090909, 0.08196721, 0.08196721 ,0.        ])

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.patches import PathPatch

# df = pd.DataFrame(np.random.rand(140, 4), columns=['A', 'B', 'C', 'D'])

# df['models'] = pd.Series(np.repeat(['model1','model2', 'model3', 'model4',     'model5', 'model6', 'model7'], 20))

# bp_dict = df.boxplot(
#     by="models",layout=(4,1),figsize=(6,8),
#     return_type='both',
#     patch_artist = True,
# )

# colors = ['b', 'y', 'm', 'c', 'g', 'b', 'r', 'k', ]
# for row_key, (ax,row) in bp_dict.iteritems():
#     ax.set_xlabel('')
#     for i,box in enumerate(row['boxes']):
#         box.set_facecolor(colors[i])

# plt.show()
# def quicksort(arr):    
#     if len(arr) <= 1:        
#         return arr    
#     pivot = arr[len(arr) // 2]    
#     left = [x for x in arr if x < pivot]    
#     middle = [x for x in arr if x == pivot]    
#     right = [x for x in arr if x > pivot]    
    
#     return quicksort(left) + middle + quicksort(right)
# print(quicksort([3, 6, 8, 19, 1, 5]))  # [1，3, 5, 6, 8, 19
import torch
import torch.nn.functional as f

a=torch.rand(3,4)
tmp=[[0],[0],[0]]
t=torch.tensor(tmp)
b=(torch.arange(1,5,device=a.device)[None,:]==t).long()

loss=torch.nn.CrossEntropyLoss()
loss1=loss(a,t.squeeze(-1))
print(loss1)
print(b)





# import matplotlib.pyplot as plt
# import numpy as np
# import torch 
# import torch 
# import torch.nn as nn

# loss = nn.CrossEntropyLoss()

# # input, NxC=2x3

# # input = torch.randn((1,2,2,2))
# # print(torch.nn.functional.softmax(input))
# # # target, N

# # target = torch.empty((1,2,2), dtype=torch.long).random_(2)
# # print(target)
# # output = loss(input, target)
# # print(output)
# tensor1=torch.randint(3,13,size=(2,2),dtype=torch.float32,requires_grad=True)
# print(tensor1.unsqueeze(dim=1).shape)
# tensor2=torch.cat((tensor1.unsqueeze(dim=1),tensor1.unsqueeze(dim=1)),dim=1)
# # tensor2.requires_grad=True
# tensor3=tensor1+tensor2
# print(torch.arange(1,10))
#output.backward()

# fig2=plt.figure()
# ax1 = fig1.add_subplot(1,1,1) # 画2行1列个图形的第1个
# ax2 = fig2.add_subplot(1,1,1) # 画2行1列个图形的第2个
# ax1.set_xlabel("efef")

# ax1.plot(np.random.randint(1,5,5), np.arange(5))
# ax2.plot(np.arange(10)*3, np.arange(10))
# fig2.savefig("dfd.png",dpi=3000,bbox_inches='tight')
# plt.show()