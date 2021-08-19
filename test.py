import argparse
import os
import collections
import re
import cv2
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import seaborn as sns
from torchvision import transforms as T
from retinanet import model
from retinanet.dataloader import CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, Normalizer,UnNormalizer
from torch.utils.data import DataLoader,Dataset
from matplotlib import pyplot as plt
from retinanet import coco_eval
from retinanet import csv_eval
import sys
from sklearn.utils import shuffle 
import csv
import time
import torch.nn as nn
assert torch.__version__.split('.')[0] == '1'

print('CUDA available: {}'.format(torch.cuda.is_available()))

#new
DIR = "global-wheat-detection/"
DIR_TRAIN = DIR + "train"
DIR_TEST = DIR + "test"
df = pd.read_csv(DIR + "test.csv")
df.head()

df['x'] = -1
df['y'] = -1
df['w'] = -1
df['h'] = -1

def expand_bbox(x):
    r = np.array(re.findall("([0-9]+[.]?[0-9]*)", x))
    if len(r) == 0:
        r = [-1, -1, -1, -1]
    return r

df[['x', 'y', 'w', 'h']] = np.stack(df['bbox'].apply(lambda x: expand_bbox(x)))
df.drop(columns=['bbox'], inplace=True)
df['x'] = df['x'].astype(np.float)
df['y'] = df['y'].astype(np.float)
df['w'] = df['w'].astype(np.float)
df['h'] = df['h'].astype(np.float)
#df['source'] = df['source'].astype(np.str)
df.head()

unq_values = df["image_id"].unique()
print("Total Records: ", len(df))
print("Unique Images: ",len(unq_values))

null_values = df.isnull().sum(axis = 0)
print("\n> Null Values in each column <")
print(null_values)

sources = df["source"].unique()
print("Total Sources: ",len(sources))
print("\n> Sources <\n",sources)

plt.figure(figsize=(14,8))
plt.title('Source Distribution', fontsize= 20)
sns.countplot(x = "source", data = df)

images = df['image_id'].unique()

test_imgs = images
test_df = df[df['image_id'].isin(test_imgs)]


print(test_imgs)

def load_classes(csv_reader):
    result = {}

    for line, row in enumerate(csv_reader):
        line += 1

        try:
            class_name, class_id = row
        except ValueError:
            raise(ValueError('line {}: format should be \'class_name,class_id\''.format(line)))
        class_id = int(class_id)

        if class_name in result:
            raise ValueError('line {}: duplicate class name: \'{}\''.format(line, class_name))
        result[class_name] = class_id
    return result

def format_prediction_string(boxes, scores):
    pred_strings = []
    for j in zip(scores, boxes):
        pred_strings.append("{0:.4f} {1} {2} {3} {4}".format(j[0], j[1][0], j[1][1], j[1][2], j[1][3]))
    return " ".join(pred_strings)

# Draws a caption above the box in an image
def draw_caption(image, box, caption):
    b = np.array(box).astype(int)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)


def detect_image(image_path, model, class_list):

    with open(class_list, 'r') as f:
        classes = load_classes(csv.reader(f, delimiter=','))

    labels = {}
    for key, value in classes.items():
        labels[value] = key
    #加载ｐｔ文件
    #model = torch.load(model_path)
    #model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    #model.load_state_dict(torch.load(model_path))
    if torch.cuda.is_available():
        model = model.cuda()

    model.training = False
    model.eval()
    results = []
     
    for img_name in os.listdir(image_path):

        image = cv2.imread(os.path.join(image_path, img_name))
        if image is None:
            continue
        image_orig = image.copy()

        rows, cols, cns = image.shape

        smallest_side = min(rows, cols)

        # rescale the image so the smallest side is min_side
        min_side = 736
        max_side = 1024
        scale = min_side / smallest_side

        # check if the largest side is now greater than max_side, which can happen
        # when images have a large aspect ratio
        largest_side = max(rows, cols)

        if largest_side * scale > max_side:
            scale = max_side / largest_side

        # resize the image with the computed scale
        image = cv2.resize(image, (int(round(cols * scale)), int(round((rows * scale)))))
        rows, cols, cns = image.shape

        pad_w = 32 - rows % 32
        pad_h = 32 - cols % 32

        new_image = np.zeros((rows + pad_w, cols + pad_h, cns)).astype(np.float32)
        new_image[:rows, :cols, :] = image.astype(np.float32)
        image = new_image.astype(np.float32)
        image /= 255
        image -= [0.485, 0.456, 0.406]
        image /= [0.229, 0.224, 0.225]
        image = np.expand_dims(image, 0)
        image = np.transpose(image, (0, 3, 1, 2))
        


        with torch.no_grad():

            image = torch.from_numpy(image)
            if torch.cuda.is_available():
                image = image.cuda()

            st = time.time()
            #print(image.shape, image_orig.shape, scale)
            #print(model(image.cuda().float()))
            scores, classification, transformed_anchors = model(image.cuda().float())
            print('Elapsed time: {}'.format(time.time() - st))
            idxs = np.where(scores.cpu() > 0.7)
            for j in range(idxs[0].shape[0]):
                bbox = transformed_anchors[idxs[0][j], :]
                
                x1 = int(bbox[0] / scale)
                y1 = int(bbox[1] / scale)
                x2 = int(bbox[2] / scale)
                y2 = int(bbox[3] / scale)
                label_name = labels[int(classification[idxs[0][j]])]
                #print(bbox, classification.shape)
                score = scores[j]
                caption = '{} {:.3f}'.format(label_name, score)
                #draw_caption(img, (x1, y1, x2, y2), label_name)
                draw_caption(image_orig, (x1, y1, x2, y2), caption)
                cv2.rectangle(image_orig, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)
            result={'image_id':img_name,'PredictionString':format_prediction_string(transformed_anchors,scores)}
            results.append(result)
            test_df = pd.DataFrame(results,columns=['image_id','PredictionString'])
            test_df.to_csv('submission.csv',index=False)
            test_df.head()
            #cv2.imshow('detections', image_orig)
            cv2.imwrite("./{}.png".format(img_name), image_orig, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])  
            cv2.waitKey(0)
            cv2.destroyAllWindows()

def plot_img(image_name):
    
    fig, ax = plt.subplots(1, 2, figsize = (10, 10))
    ax = ax.flatten()
    
    records = df[df['image_id'] == image_name]
    img_path = os.path.join(DIR_TRAIN, image_name + ".jpg")
    
    image = cv2.imread(img_path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
    image /= 255.0
    image2 = image
    
    ax[0].set_title('Original Image')
    ax[0].imshow(image)
    
    for idx, row in records.iterrows():
        box = row[['x', 'y', 'w', 'h']].values
        xmin = box[0]
        ymin = box[1]
        width = box[2]
        height = box[3]
        
        cv2.rectangle(image2, (int(xmin),int(ymin)), (int(xmin + width),int(ymin + height)), (255,0,0), 3)
    
    ax[1].set_title('Image with Bondary Box')
    ax[1].imshow(image2)

    plt.show()


class GWD(Dataset):

    def __init__(self, dataframe, image_dir, class_list, mode = "train", transforms = None):
        
        super().__init__()
        self.image_ids = dataframe['image_id'].unique()
        self.df = dataframe
        self.image_dir = image_dir
        self.mode = mode
        self.transforms = transforms
        self.class_list = class_list
        try:
            with self._open_for_csv(self.class_list) as file:
                self.classes = self.load_classes(csv.reader(file, delimiter=','))
        except ValueError as e:
            raise(ValueError('invalid CSV class file: {}: {}'.format(self.class_list, e)))
        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

    def name_to_label(self, name):
        return self.classes[name]

    def _open_for_csv(self, path):
        """
        Open a file with flags suitable for csv.reader.
        This is different for python2 it means with mode 'rb',
        for python3 this means 'r' with "universal newlines".
        """
        if sys.version_info[0] < 3:
            return open(path, 'rb')
        else:
            return open(path, 'r', newline='')

    def data_records(self,records):
        result = {}
        #print(records)
        for idx, row in records.iterrows():
                
                box = row[['x', 'y', 'w', 'h','source','image_id']].values
                #print('row',box[0])
                x1 = box[0]
                y1 = box[1]
                x2 = box[0] + box[2]
                y2 = box[1] + box[3]
                class_name = box[4]
                image_id = box[5]
                #print('id',image_id)
                #print('re',result)
                if image_id not in result:
                    result[image_id] = []
                result[image_id].append({'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2, 'class': class_name})
                #print('result',result)
        return result

    def load_annotations(self,index: int):
        labels = {'usask_1':0,'arvalis_1':1,'inrae_1':2,'ethz_1':3,'arvalis_3':4,'rres_1':5,'arvalis_2':6}
        ar = []
        image_id = self.image_ids[index]
        records = self.df[self.df['image_id'] == image_id]
        annotation_list = self.data_records(records)
        annotation_list = annotation_list[image_id]
        #print('rererereerer',annotation_list)
        annotations = np.zeros((0, 5))
        if len(annotation_list) == 0:
            return annotations

        # parse annotations
        for idx, a in enumerate(annotation_list):
            #print('aaaaaaaaaa',a)
            # some annotations have basically no width / height, skip them
            x1 = a['x1']
            x2 = a['x2']
            y1 = a['y1']
            y2 = a['y2']
            annotation        = np.zeros((1, 5))
            
            annotation[0, 0] = x1
            annotation[0, 1] = y1
            annotation[0, 2] = x2
            annotation[0, 3] = y2
            ar = a['class']
            annotation[0, 4]  = labels[ar]
            #print('a4',annotation)
            annotations       = np.append(annotations, annotation, axis=0)
        #annotations[:, 0:4] = records[['x', 'y', 'w', 'h']].values
        #annotations[:, 2] = annotations[:, 0] + annotations[:, 2]
        #annotations[:, 3] = annotations[:, 1] + annotations[:, 3]
        #print(records[['source']].values)
        #ar = records[['source']].values
        #for i in range(len(ar)):
                #ar[i] = labels[ar[i,0]]
        #print('4',ar)
        #annotations[:, 4] = ar # This is for label, as we have only 1 class, it is always 1
            #print('load',annotations)
        return annotations

    def _parse(self, value, function, fmt):
        """
        Parse a string into a value, and format a nice ValueError if it fails.
        Returns `function(value)`.
        Any `ValueError` raised is catched and a new `ValueError` is raised
        with message `fmt.format(e)`, where `e` is the caught `ValueError`.
        """
        try:
            return function(value)
        except ValueError as e:
            raise_from(ValueError(fmt.format(e)), None)

    def num_classes(self):
        return max(self.classes.values()) + 1

    def name_to_label(self, name):
        return self.classes[name]


    def load_classes(self, csv_reader):
        result = {}

        for line, row in enumerate(csv_reader):
            line += 1

            try:
                class_name, class_id = row
            except ValueError:
                raise(ValueError('line {}: format should be \'class_name,class_id\''.format(line)))
            class_id = self._parse(class_id, int, 'line {}: malformed class ID: {{}}'.format(line))

            if class_name in result:
                raise ValueError('line {}: duplicate class name: \'{}\''.format(line, class_name))
            result[class_name] = class_id
        return result


    def label_to_name(self, label):
        return self.labels[label]

    def __getitem__(self, index: int):

        # Retriving image id and records from df
        image_id = self.image_ids[index]
        records = self.df[self.df['image_id'] == image_id]

        # Loading Image
        image = cv2.imread(f'{self.image_dir}/{image_id}.jpg', cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0
        labels = {'usask_1':0,'arvalis_1':1,'inrae_1':2,'ethz_1':3,'arvalis_3':4,'rres_1':5,'arvalis_2':6}
        arr = []
        # If mode is set to train, then only we create targets
        if self.mode == "train" or self.mode == "valid":

            # Converting xmin, ymin, w, h to x1, y1, x2, y2
            boxes = np.zeros((records.shape[0], 5))
            boxes[:, 0:4] = records[['x', 'y', 'w', 'h']].values
            boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
            boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
            ar = records[['source']].values
            #print(ar[1,0])
            for i in range(len(ar)):
                ar[i] = labels[ar[i,0]]
            #print(ar)
            boxes[:, 4] = ar[:,0] # This is for label, as we have only 1 class, it is always 1
            #print(boxes)
            # Applying Transforms
            sample = {'img': image, 'annot': boxes}

            if self.transforms:
                sample = self.transforms(sample)

            return sample
        
        elif self.mode == "test":
            
            # We just need to apply transoforms and return image
            if self.transforms:
                
                sample = {'img' : image}
                sample = self.transforms(sample)
                
            return sample
        

    def __len__(self) -> int:
        return self.image_ids.shape[0]

class ClassificationModel(nn.Module):
    def __init__(self, num_features_in, num_anchors=9, num_classes=80, prior=0.01, feature_size=256):
        super(ClassificationModel, self).__init__()

        self.num_classes = num_classes
        self.num_anchors = num_anchors

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.output = nn.Conv2d(feature_size, num_anchors * num_classes, kernel_size=3, padding=1)
        self.output_act = nn.Sigmoid()

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        out = self.act4(out)

        out = self.output(out)
        out = self.output_act(out)

        # out is B x C x W x H, with C = n_classes + n_anchors
        out1 = out.permute(0, 2, 3, 1)

        batch_size, width, height, channels = out1.shape

        out2 = out1.view(batch_size, width, height, self.num_anchors, self.num_classes)

        return out2.contiguous().view(x.shape[0], -1, self.num_classes)

def main(args=None):
	class_list = 'data/class_list.csv'
	
	test_dataset = GWD(test_df, DIR_TRAIN, class_list, mode = "train", transforms = T.Compose([Normalizer(), Resizer()]))

	test_data_loader = DataLoader(test_dataset,batch_size = 1,shuffle = True,num_workers = 4,collate_fn = collater)

	#retinanet = model.resnest101(num_classes=7, pretrained=True)
	retinanet = torch.load('dsa-bifpn49.pt')
	mAP = csv_eval.evaluate(test_dataset, retinanet)

	image_dir = 'global-wheat-detection/train'
	detect_image(DIR_TRAIN, retinanet, class_list)


if __name__ == '__main__':
	main()
