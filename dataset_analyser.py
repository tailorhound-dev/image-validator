import pandas as pd
import requests
from io import BytesIO
from rembg import remove
from PIL import Image
import torch
import torchvision
from torchvision import transforms as T
import numpy as np
import analysis_functions as analysis

#Parser to extract all information relevent to training 'naive pants' model

#Database columns = 'id', 'source', 'gender', 'height', 'bust', 'cup', 'waist', 'hips', 'dress', 'chest', 'inseam', 'collar', 'suitlength', 'photourl', 'fullheight', 'ideal', 'dead', 'exclude'
data_frame = pd.read_csv('csv/photos_1.csv')
naive_pants_data_frame = data_frame.drop(columns=['source', 'dress', 'suitlength', 'exclude'])

#Reset bodypix fullheight calc and define new schema
naive_pants_data_frame['fullheight'].replace(1, 0, inplace=True)
naive_pants_data_frame = naive_pants_data_frame.rename(columns={'fullheight': 'solo'})
naive_pants_data_frame = naive_pants_data_frame.rename(columns={'ideal': 'fullbody'})
naive_pants_data_frame['standing'] = 0 
naive_pants_data_frame['facing_straight'] = 0
naive_pants_data_frame['ideal'] = 0


#Prepare model for keypoint analysis (checking pose)
model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True, weights='KeypointRCNN_ResNet50_FPN_Weights.DEFAULT')
model.eval()
model.to(torch.device("cuda:0"))

#Download and process images check for fullheight bool, save resulting images and criteria
for index, row in naive_pants_data_frame.iterrows():
        
        #Segment image
        img = Image.open(BytesIO(requests.get(row['photourl']).content))
        no_bg = remove(img)
        width, height = img.size

        #Create high contrast RGB variant of no_bg
        new_bg = Image.new(mode="RGB", size = (height, height), color = (0, 177, 64))
        width_added = height - width  
        offset = width_added / 4
        new_bg.paste(no_bg, (int(offset), 0), no_bg)
        
        #Prepare the image for inference
        transform = T.Compose([T.ToTensor()])
        new_bg_tensor = transform(new_bg)
        new_bg_tensor = new_bg_tensor.cuda()
        
        #Run inference
        output = model([new_bg_tensor])[0]

        #Analyse result
        results = analysis.analyse_entry(index, no_bg, width, height, output)

        if results[0] == 1:
            row['solo'] = 1
        if results[1] == 1:
            row['fullbody'] = 1
        if results[2] == 1:
            row['standing'] = 1
        if results[3] == 1:
            row['facing_straight'] = 1

        if sum(results) == 0:
            print("Multiple / No subjects")
            path = "dataset/images/incorrect/"  + str(index) + ".jpg"
            new_bg.save(path)

        if results == [1, 0, 0, 0]:
            print("Solo but not fullbody")
            path = "dataset/images/solo/"+ str(index) + ".jpg"
            new_bg.save(path)

        if results == [1, 1, 0, 0]: 
            print("Fullbody but not facing")
            path = "dataset/images/fullbody/" + str(index) + ".jpg"
            new_bg.save(path)
        
        if results == [1, 1, 1, 0]:
            print("Standing but not facing")
            path = "dataset/images/standing/" + str(index) + ".jpg"
            new_bg.save(path)
        
        if results == [1, 1, 0, 1]:
            print("Facing but not standing")
            path = "dataset/images/facing/" + str(index) + ".jpg"
            new_bg.save(path)

        if results == [1, 1, 1, 1]:
            print("Standing and facing")
            row['ideal'] = 1
            path = "dataset/images/ideal/" + str(index) + ".jpg"
            new_bg.save(path)


naive_pants_data_frame.to_csv('./dataset/annotations/dataset.csv')





