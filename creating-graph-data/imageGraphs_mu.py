## importing libraries
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from tqdm.auto import tqdm
import csv
import pandas as pd
import os
from transformers import ViTImageProcessor, ViTModel
import matplotlib.pyplot as plt
import random
from PIL import Image, ImageDraw
import timm
# from transformers import BeitModel, BeitImageProcessor
import timm.data
from transformers import MobileViTModel, MobileViTImageProcessor

def random_color():
    return tuple([random.randint(0, 255) for _ in range(3)])

def display_image_with_boxes(image, boxes):
    draw = ImageDraw.Draw(image)
    for box in boxes:
        color = random_color()
        draw.rectangle(((box[0], box[1]), (box[2], box[3])), outline=color, width=3)
    plt.imshow(image)
    plt.axis('off')
    plt.show()

def readcsv(fileName):
    '''
    将所有对话的目标语句、上下文语句、讽刺标签、情感标签、情绪标签都集合到uttDict中
    目标语句和上下文语句都在utterance list中，以对话的时间顺序展开，最后是目标语句
    现在返回的utterance是所有的utt，最多的语句是12句，最少的语句是2句
    '''
    with open(fileName, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        uttNameList = []
        # context = np.array()
        for i in reader:
            if i[0] != '':
                uttNameList.append(i[0])
        uttNameList = list(set(uttNameList))
        uttNameList.remove("KEY")
        uttDict = {}
        for name in uttNameList:
            uttDict[name] = {}
            uttDict[name]['utterance'] = []
            uttDict[name]['sarcasm-label'] = []
            uttDict[name]['sentiment-label'] = []
            uttDict[name]['emotion-label'] = []
            uttDict[name]['utt-number'] = ''

    with open(fileName, 'r', encoding='utf-8') as f1:
        reader = csv.reader(f1)
        for item in reader:
            if item[0] == 'KEY' or item[0] == '':
                continue
            uttDict[item[0]]['sarcasm-label'].append(item[4])
            uttDict[item[0]]['sentiment-label'].append(item[5])
            uttDict[item[0]]['emotion-label'].append(item[7])
            uttDict[item[0]]['utterance'].append(item[2])
            uttDict[item[0]]['utt-number'] = item[0]
    return uttDict, uttNameList


def store_features(od_model, processor, feature_model, uttNameList, uttDict, root_dir, image_dir, store_dir):
    lengths = []

    for idx in range(len(uttNameList)):
        uttName = uttNameList[idx]
        text = list(uttDict[uttName]['utterance'])
        print('uttName:', uttName)
        print(len(text))

        '''
        process the image feature
        '''

        frame_output_dir = f"{root_dir}{image_dir}{uttName}"
        frameFile = os.listdir(frame_output_dir)
        numFrames = len(frameFile)
        sampleSeq = np.linspace(1, numFrames, num=len(text), endpoint=True, dtype=int)

        def generateFrameName(sampleSeq):
            return [f"{seq:05d}.jpg" for seq in sampleSeq]

        frameNames = generateFrameName(sampleSeq)

        for j, frame in enumerate(frameNames):
            img_path = os.path.join(frame_output_dir, frame)
            try:
                image = Image.open(img_path).convert('RGB')
            except:
                print("Image name:", img_path)
                continue

            image_t = image_transform(image).to(device)
            image_t = image_t.unsqueeze(0)  # add a batch dimension
            image_t = image_t[:, :3, :, :]

            with torch.no_grad():
                outputs = od_model(image_t)  # get the predictions on the image

            # Extract bounding boxes
            boxes = outputs[0]['boxes'].cpu().numpy()

            # Display image with bounding boxes
            # display_image_with_boxes(image.copy(), boxes)

            image_features = []

            transformed_image = image_transform(image).unsqueeze(0)
            transformed_image = transformed_image.to(device)
            transformed_image = transformed_image[:, :3, :, :]

            # Process the whole image using BEiT
            inputs = processor(images=image, return_tensors="pt").to(device)

            with torch.no_grad():
                feature = feature_model(**inputs)
            # print(type(feature), feature.keys())

            ## Find features for the whole image
            feature = feature.last_hidden_state[:, 0]

            ## Save whole image representation
            # 检查_full_image.npy文件是否存在
            filename = f'{root_dir}{store_dir}{uttName}_{j}_full_image.npy'
            if os.path.exists(filename):
                print(f"File {filename} already exists. Skipping.")
                continue
            else:
                np.save(filename, feature.cpu().numpy())

            ## Loop over the bounding boxes detected
            for i in range(len(outputs[0]['boxes'])):
                box = outputs[0]['boxes'][i]

                ## create a cropped image
                x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                cropped_image = image.convert("RGB").crop((x1, y1, x2, y2))

                ## Transform the cropped image
                transformed_image = image_transform(cropped_image).unsqueeze(0)
                transformed_image = transformed_image.to(device)
                transformed_image = transformed_image[:, :3, :, :]

                ## Find feature for each object
                # with torch.no_grad():
                #     feature = feature_model(transformed_image)
                # Process the cropped image using BEiT
                inputs = processor(images=cropped_image, return_tensors="pt").to(device)

                with torch.no_grad():
                    feature = feature_model(**inputs)
                feature = feature.last_hidden_state[:, 0]

                image_features.append(feature.detach().cpu().numpy())

            lengths.append(len(image_features))

            ### Store object level representations
            try:
                image_features = np.concatenate(image_features, axis=0)
            except:
                print("Image name:", img_path)
                continue

            filename = f'{root_dir}{store_dir}{uttName}_{j}.npy'
            if os.path.exists(filename):
                print(f"File {filename} already exists. Skipping.")
                continue
            else:
                np.save(filename, image_features)

    return lengths


if __name__ == '__main__':
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    ## Load object detection model
    fatser_rcnn = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    ## Set the transforms for the images
    image_transform = transforms.Compose(
        [
            transforms.Resize(size=(224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
    )
    fatser_rcnn.eval().to(device)

    ## Load the model for feature representations
    # processor = ViTImageProcessor.from_pretrained("/home/zxl/MultiTask classification/proposed/vit-base-patch16-224/")
    # model = ViTModel.from_pretrained("/home/zxl/MultiTask classification/proposed/vit-base-patch16-224/")
    # processor = BeitImageProcessor.from_pretrained("/home/zxl/MultiTask classification/proposed/beit-base-patch16-224-pt22k-ft22k/")
    # model = BeitModel.from_pretrained("/home/zxl/MultiTask classification/proposed/beit-base-patch16-224-pt22k-ft22k/")

    processor = ViTImageProcessor.from_pretrained("/home/zxl/MultiTask classification/proposed/vit-base-patch16-224/")
    model = ViTModel.from_pretrained("/home/zxl/MultiTask classification/proposed/vit-base-patch16-224/")
    model.eval().to(device)

    ## Base directory for the data
    root_dir = "/home/zxl/MultiTask classification/"

    ## Directory location for train and test images
    # images_train = "data/memotion_dataset_7k/images/"
    image = "MultiTask-Classfication/context/"
    # images_test = ""

    ## Create a list of names for the images
    images_list = os.listdir(f'{root_dir}{image}')
    # images_list_test = os.listdir(f'{root_dir}{images_test}')

    ## Directory to store the node embeddings for each image
    store_dir = "proposed/test2/large-model/mustard/imagefeature/"

    ## File locations
    train_csv_name = "/home/zxl/MultiTask classification/M2Seq2Seq-master/mustard-dataset-train.csv"
    test_csv_name = "/home/zxl/MultiTask classification/M2Seq2Seq-master/mustard-dataset-test.csv"
    dev_csv_name = "/home/zxl/MultiTask classification/M2Seq2Seq-master/mustard-dataset-dev.csv"

    uttDict_train, uttNameList_train = readcsv(train_csv_name)
    ## store graph data for training images
    lengths = store_features(fatser_rcnn, processor, model, uttNameList_train, uttDict_train, root_dir, image, store_dir)
    print(len(lengths))

    uttDict_test, uttNameList_test = readcsv(test_csv_name)
    lengths = store_features(fatser_rcnn, processor, model, uttNameList_test, uttDict_test, root_dir, image, store_dir)
    print(len(lengths))

    uttDict_dev, uttNameList_dev = readcsv(dev_csv_name)
    lengths = store_features(fatser_rcnn, processor, model, uttNameList_dev, uttDict_dev, root_dir, image, store_dir)
    print(len(lengths))
