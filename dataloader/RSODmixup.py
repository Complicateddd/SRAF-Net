
import torch
import xml.etree.ElementTree as ET
import os
import cv2
import numpy as np
from torchvision import transforms

class RSODmixupDataset(torch.utils.data.Dataset):
    # CLASSES_NAME = (
    #     "__background__ ",
    #     "aeroplane",
    #     "bicycle",
    #     "bird",
    #     "boat",
    #     "bottle",
    #     "bus",
    #     "car",
    #     "cat",
    #     "chair",
    #     "cow",
    #     "diningtable",
    #     "dog",
    #     "horse",
    #     "motorbike",
    #     "person",
    #     "pottedplant",
    #     "sheep",
    #     "sofa",
    #     "train",
    #     "tvmonitor",
    # )
    CLASSES_NAME = (
        "__background__ ",
    'aircraft','oiltank','overpass','playground')
    def __init__(self,root_dir,resize_size=[800,1024],split='trainval',use_difficult=False):
        self.root=root_dir
        self.use_difficult=use_difficult
        self.imgset=split

        self._annopath = os.path.join(self.root, "Annotations", "%s.xml")
        self._imgpath = os.path.join(self.root, "JPEGImages", "%s.jpg")
        self._imgsetpath = os.path.join(self.root, "ImageSets", "Main", "%s.txt")
        # self._img_scenepath=os.path.join(self.root,"scene_label","%s.txt")
        self._img_muti_scenepath=os.path.join(self.root,"muti_scene_label","%s.txt")
        with open(self._imgsetpath%self.imgset) as f:
            self.img_ids=f.readlines()
        self.img_ids=[x.strip() for x in self.img_ids]
        self.name2id=dict(zip(RSODmixupDataset.CLASSES_NAME,range(len(RSODmixupDataset.CLASSES_NAME))))
        self.resize_size=resize_size
        self.mean=[0.485, 0.456, 0.406]
        self.std=[0.229, 0.224, 0.225]
        print("INFO=====>voc dataset init finished  ! !")

    def __len__(self):
        return len(self.img_ids)

    def _read_img_rgb(self,path):
        return cv2.cvtColor(cv2.imread(path),cv2.COLOR_BGR2RGB)

    def __getitem__(self,index):
        
        img_id=self.img_ids[index]
        img=self._read_img_rgb(self._imgpath%img_id)
        # print(type(img))
        # print(self._imgpath%img_id)

        # scene_name=self._img_scenepath%(img_id+".jpg")
        # with open(scene_name) as f:
        #     scene_class=f.readlines()
        # scene_class=[int(scene_class[0])-1]
        # scene_class=[int(scene_class[0])-1]
        muti_scene_label_name=self._img_muti_scenepath%(img_id+".jpg")
        muti_scene_list=[0.]*4
        with open(muti_scene_label_name) as f:
            for s in f.readlines():
                # print(s[0])
                muti_scene_list[int(s[0])]=1.
        
        anno=ET.parse(self._annopath%img_id).getroot()
        boxes=[]
        classes=[]
        for obj in anno.iter("object"):
            # difficult = int(obj.find("difficult").text) == 1
            # if not self.use_difficult and difficult:
            #     continue
            _box=obj.find("bndbox")
            # Make pixel indexes 0-based
            # Refer to "https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/datasets/pascal_voc.py#L208-L211"
            box=[
                _box.find("xmin").text, 
                _box.find("ymin").text, 
                _box.find("xmax").text, 
                _box.find("ymax").text,
            ]
            TO_REMOVE=1
            box = tuple(
                map(lambda x: x - TO_REMOVE, list(map(float, box)))
            )
            boxes.append(box)

            name=obj.find("name").text.lower().strip()
            classes.append(self.name2id[name])
        
        boxes=np.array(boxes,dtype=np.float32)
        img,boxes=self.preprocess_img_boxes(img,boxes,self.resize_size)
        # print(img)
        # img=transforms.ToTensor()(img)

        img=torch.tensor(img)
        img=img.transpose(1,2).transpose(0,1).contiguous()
        # print("++++++++++",img)
        # img=img.double()
        # print(img)
        boxes=torch.from_numpy(boxes)
        classes=torch.LongTensor(classes)
        muti_scene_class=torch.tensor(muti_scene_list)
        # scene_class=torch.LongTensor(scene_class)
        return img,boxes,classes,muti_scene_class
        

    def preprocess_img_boxes(self,image,boxes,input_ksize):
        '''
        resize image and bboxes 
        Returns
        image_paded: input_ksize  
        bboxes: [None,4]
        '''
        min_side, max_side    = input_ksize
        h,  w, _  = image.shape
        # print(image.shape)
        smallest_side = min(w,h)
        largest_side=max(w,h)
        scale=min_side/smallest_side
        if largest_side*scale>max_side:
            scale=max_side/largest_side
        nw, nh  = int(scale * w), int(scale * h)
        # print(scale)
        # print(nw,nh)
        image_resized = cv2.resize(image, (nw, nh))
        # print(image_resized.shape)
        pad_w=32-nw%32
        pad_h=32-nh%32
        # print("padwh",pad_w,pad_h)
        image_paded = np.zeros(shape=[nh+pad_h, nw+pad_w, 3],dtype=np.float32)
        image_paded[:nh, :nw, :] = image_resized
        # print(image_paded.shape)

        if boxes is None:
            return image_paded
        else:
            boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale 
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale 
            return image_paded, boxes
    @staticmethod
    def mixup(input_images,input_bbox,input_target_class,input_scene_class,bate=0.5):
        '''input_images:(tensor) [b,c,h,w]
        input_bbox: (list) length=b :  [tensor1 [m1,4],tensor2 [m2,4]..]
        input_input_target_class: (list) length=b  [tensor1 [m,1],tensor2 [m,1]..]
        input_scene_class: (list) length=b [tensor1 [classnum],tensor2 [classnum]..]'''
        # print(input_images)
        batch_size=input_images.shape[0]
        rp1 = torch.randperm(batch_size)
        rp2 = torch.randperm(batch_size)
        # print(rp1)
        # print(rp2)

        inputs1 = input_images[rp1]
        inputs2 = input_images[rp2]
        b=torch.tensor(bate).float()
        inputs_shuffle_image=inputs1.float()*(b)+inputs2.float()*(1-b)
        
        rp1_list=rp1.tolist()
        rp2_list=rp2.tolist()
        
        # input_images_rp1=[input_images[i] for i in rp1_list]
        # input_images_rp2=[input_images[i] for i in rp2_list]


        input_bbox_rp1=[input_bbox[i] for i in rp1_list]
        input_bbox_rp2=[input_bbox[i] for i in rp2_list]
        input_target_class_rp1=[input_target_class[i] for i in rp1_list]
        input_target_class_rp2=[input_target_class[i] for i in rp2_list]
        input_scene_class_rp1=[input_scene_class[i] for i in rp1_list]
        input_scene_class_rp2=[input_scene_class[i] for i in rp2_list]

        # input_images_new=[]
        input_bbox_new=[]
        input_class_new=[]
        input_scene_class_new=[]

        index_for_equal=(rp1!=rp2).tolist()
        # print(index_for_equal)
        for i in range(batch_size):
            if index_for_equal[i]==0:
                input_bbox_new.append(input_bbox_rp1[i])
                input_class_new.append(input_target_class_rp1[i])
                input_scene_class_new.append(input_scene_class_rp1[i])
                # input_images_new.append(input_scene_class_rp1[i])
            else:
                # temp_image=input_scene_class_rp1[i]+input_scene_class_rp2[i]
                temp_bbox=torch.cat((input_bbox_rp1[i],input_bbox_rp2[i]))
                temp_class=torch.cat((input_target_class_rp1[i],input_target_class_rp2[i]))
                temp_scene_class=input_scene_class_rp1[i]*b+input_scene_class_rp2[i]*(1-b)
                a=torch.ones_like(temp_scene_class)
                temp_scene_class=torch.where(temp_scene_class>1,a,temp_scene_class)
                # input_images_new.append(temp_image)
                input_bbox_new.append(temp_bbox)
                input_class_new.append(temp_class)
                input_scene_class_new.append(temp_scene_class)
        
        return inputs_shuffle_image,input_bbox_new,input_class_new,input_scene_class_new


    def collate_fn(self,data):
        # imgs_list,boxes_list,classes_list,scene_list=zip(*data)
        imgs_list,boxes_list,classes_list,muti_scene_class=zip(*data)
        assert len(imgs_list)==len(boxes_list)==len(classes_list)
        batch_size=len(boxes_list)
        pad_imgs_list=[]
        pad_boxes_list=[]
        pad_classes_list=[]

        h_list = [int(s.shape[1]) for s in imgs_list]
        w_list = [int(s.shape[2]) for s in imgs_list]
        max_h = np.array(h_list).max()
        max_w = np.array(w_list).max()
        for i in range(batch_size):
            img=imgs_list[i]
            temp=torch.nn.functional.pad(img,(0,int(max_w-img.shape[2]),0,int(max_h-img.shape[1])),value=0.)
            temp=temp.float()
            pad_imgs_list.append(temp)
        batch_imgs_tensor=torch.stack(pad_imgs_list)
        batch_imgs_new,boxes_list_new,classes_list,muti_scene_class=self.mixup(batch_imgs_tensor,boxes_list,classes_list,muti_scene_class)
        mean=torch.tensor([0.485, 0.456, 0.406])
        std=torch.tensor([0.229, 0.224, 0.225])
        batch_imgs_new=(batch_imgs_new.transpose(1,2).transpose(2,3).contiguous()/255.-mean)/std
        batch_imgs_new=batch_imgs_new.transpose(2,3).transpose(1,2).contiguous()
        max_num=0
        for i in range(batch_size):

            n=boxes_list_new[i].shape[0]
            if n>max_num:max_num=n   
        for i in range(batch_size):
            pad_boxes_list.append(torch.nn.functional.pad(boxes_list_new[i],(0,0,0,max_num-boxes_list_new[i].shape[0]),value=-1))
            pad_classes_list.append(torch.nn.functional.pad(classes_list[i],(0,max_num-classes_list[i].shape[0]),value=-1))
        
        batch_imgs_tensor=torch.stack(pad_imgs_list)
        batch_boxes=torch.stack(pad_boxes_list)
        batch_classes=torch.stack(pad_classes_list)
        batch_muti_scene_class=torch.stack(muti_scene_class)
        return batch_imgs_new.float(),batch_boxes,batch_classes,batch_muti_scene_class