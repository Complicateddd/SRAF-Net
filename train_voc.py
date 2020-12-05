'''
@Author: xxxmy
@Github: github.com/VectXmy
@Date: 2019-09-26
@Email: xxxmy@foxmail.com
'''
from model.fcos import FCOSDetector
import torch
from dataloader.VOC_dataset import VOCDataset
import math,time
from dataloader.RSOD import RSODDataset
from dataloader.NW import NWDataset
from dataloader.DIOR import DIORDataset
from dataloader.DIORmixup import DIORmixupDataset
from dataloader.NWmixup import NWmixupDataset
from dataloader.RSODmixup import RSODmixupDataset
# from torch.utils.tensorboard import SummaryWriter
from model.config import DefaultConfig

config=DefaultConfig
if config.class_num==4:
    if config.mixup:
        print('mixup')
        train_dataset=RSODmixupDataset("/home/ubuntu/dataset/RSOD",resize_size=[800,1024],split='train')
        # val_dataset=NWmixupDataset("/home/ubuntu/dataset/nwpuvhr10",resize_size=[800,1024],split='val')
    else:
        train_dataset=RSODDataset("/home/ubuntu/dataset/RSOD",resize_size=[800,1024],split='train')
        # val_dataset=RSODDataset("/home/ubuntu/dataset/RSOD",resize_size=[800,1024],split='val')
elif config.class_num==10:
    if config.mixup:
        print('mixup')
        train_dataset=NWmixupDataset("/home/ubuntu/dataset/nwpuvhr10",resize_size=[800,1024],split='train')
    else:
        train_dataset=NWDataset("/home/ubuntu/dataset/nwpuvhr10",resize_size=[800,1024],split='train')
        # val_dataset=NWDataset("/home/ubuntu/dataset/nwpuvhr10",resize_size=[800,1024],split='val')
elif config.class_num==20:
    if config.mixup:
        print('mixup')
        train_dataset=DIORmixupDataset("/home/ubuntu/dataset/DIOR",resize_size=[800,1024],split='train')
    else:
        train_dataset=DIORDataset("/home/ubuntu/dataset/DIOR",resize_size=[800,1024],split='train')
    # val_dataset=DIORDataset("/home/ubuntu/dataset/DIOR",resize_size=[800,1024],split='val')

model=FCOSDetector(mode="training").cuda()

# 



# lambda1=lambda epoch: (epoch / 520) if epoch < 520 else 0.5 * (math.cos((epoch - 520)/(200 * 65 - 520) * math.pi) + 1)
# optimizer = optim.SGD(model.parameters(), lr=lr_rate, momentum=0.9, nesterov=True)


# scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

if config.muti_lr:
    Scene_params = list(map(id,model.fcos_body.scene_head.parameters()))
    backbone_params=list(map(id,model.fcos_body.backbone.parameters()))
    base_params = filter(lambda p: id(p) not in Scene_params and id(p) not in backbone_params,
        model.parameters())
    
    params = [{"params":model.fcos_body.scene_head.parameters(), "lr":1e-4},
    {"params":base_params, "lr":1e-3},
    {"params":model.fcos_body.backbone.parameters(), "lr":1e-4}]
    optimizer=torch.optim.Adam(params)

else:
    print("simple lr")
    lr_rate=1e-4
    optimizer=torch.optim.Adam(model.parameters(),lr=lr_rate)
model=torch.nn.DataParallel(model)
# model.load_state_dict(torch.load("shidudi1e4Diormixupmuti_label_simgoid0.4_0.0001_50_C3_C4_C5_conc_max_pooling_192fcos_8001024_epoch50_loss0.2139.pth"))



# optimizer=torch.optim.SGD(model.parameters(),lr=1e-5)
BATCH_SIZE=4
EPOCHS=50
WARMPUP_STEPS_RATIO=0.12
train_loader=torch.utils.data.DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True,collate_fn=train_dataset.collate_fn)
# val_loader=torch.utils.data.DataLoader(val_dataset,batch_size=BATCH_SIZE,shuffle=True,collate_fn=val_dataset.collate_fn)
steps_per_epoch=len(train_dataset)//BATCH_SIZE
TOTAL_STEPS=steps_per_epoch*EPOCHS
WARMPUP_STEPS=TOTAL_STEPS*WARMPUP_STEPS_RATIO

GLOBAL_STEPS=1
LR_INIT=5e-5
LR_END=1e-6

# writer=SummaryWriter(log_dir="./logs")

def lr_func():
    if GLOBAL_STEPS<WARMPUP_STEPS:
        lr=GLOBAL_STEPS/WARMPUP_STEPS*LR_INIT
    else:
        lr=LR_END+0.5*(LR_INIT-LR_END)*(
            (1+math.cos((GLOBAL_STEPS-WARMPUP_STEPS)/(TOTAL_STEPS-WARMPUP_STEPS)*math.pi))
        )
    return float(lr)
scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer,[40,50],0.1)
model.train()

for epoch in range(EPOCHS):
    scheduler.step()
    for epoch_step,data in enumerate(train_loader):

        batch_imgs,batch_boxes,batch_classes,C=data
        # print(batch_imgs)
        batch_imgs=batch_imgs.cuda()
        batch_boxes=batch_boxes.cuda()
        batch_classes=batch_classes.cuda()
        batch_scene_classes=C.cuda()
        # print(batch_scene_classes.shape)
        # lr=lr_func()
        # for param in optimizer.param_groups:
        #     param['lr']=lr
        
        start_time=time.time()

        optimizer.zero_grad()
        losses,add_loss=model([batch_imgs,batch_boxes,batch_classes,batch_scene_classes])
        # if epoch<=15:
        
            # loss=add_loss.mean()
        # else:
        loss=losses[-1].mean()+add_loss.mean()
        # loss=add_loss.mean()
        loss.backward()
        optimizer.step()

        # scheduler.step()
        # print(add_loss.mean())
        end_time=time.time()
        cost_time=int((end_time-start_time)*1000)

        print("global_steps:%d epoch:%d steps:%d/%d tol_loss:%.4f cls_loss:%.4f cnt_loss:%.4f reg_loss:%.4f add_loss:%.4f cost_time:%dms lr=%.4e"%\
            (GLOBAL_STEPS,epoch+1,epoch_step+1,steps_per_epoch,loss,losses[0].mean(),losses[1].mean(),losses[2].mean(),add_loss.mean(),cost_time,scheduler.get_lr()[0]))
        
        # writer.add_scalar("loss/cls_loss",losses[0],global_step=GLOBAL_STEPS)
        # writer.add_scalar("loss/cnt_loss",losses[1],global_step=GLOBAL_STEPS)
        # writer.add_scalar("loss/reg_loss",losses[2],global_step=GLOBAL_STEPS)
        # writer.add_scalar("lr",lr,global_step=GLOBAL_STEPS)

        GLOBAL_STEPS+=1
    
    if (epoch+1)%50==0:

        torch.save(model.state_dict(),"0.8.1Diormixupmuti_label_simgoid0.4_0.0001_50_C3_C4_C5_conc_max_pooling_192fcos_8001024_epoch%d_loss%.4f.pth"%(epoch+1,loss.item()))
    






