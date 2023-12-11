import os
import cv2
import numpy as np
import darknet

import matplotlib.pyplot as plt
from utils import number_config, license_config, data_number, data_license, number_weights, license_weights, thresh
from PIL import Image, ImageDraw, ImageFont


font = ImageFont.truetype('arial.ttf',30) # font size
network_size = 416

def image_detection(image_or_path, network, class_names, class_colors, thresh):
    # Darknet doesn't accept numpy images.
    # Create one with image we reuse for each detect
    width = darknet.network_width(network)
    height = darknet.network_height(network)
    darknet_image = darknet.make_image(width, height, 3)

    if isinstance(image_or_path, str):
        image = cv2.imread(image_or_path)
    else:
        image = image_or_path
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (width, height),
                               interpolation=cv2.INTER_LINEAR)

    darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())
    detections = darknet.detect_image(network, class_names, darknet_image, thresh=thresh)

    return detections


def read_license_plate(size,detection_results):
    if size[0]<size[1]:
        sorted_data = sorted(detection_results, key=lambda x: x[2][0])
    else:
        sorted_data = sorted(detection_results, key=lambda x: x[2][1])
    result = ''.join([element[0] for element in sorted_data]) 
    accs = [float(element[1]) for element in sorted_data]
    srednia = sum(accs) / len(accs)
    return result, srednia
    

def check_input_type(input):
    video_ex = ['MOV','MP4','AVI']
    file_name, extension = os.path.splitext(input)
    extension_only = extension[1:]
    if extension_only.upper() in video_ex :
        return 'video'
    else:
        return 'image' 

def main():

    index = 0
    network_plate, class_names_plate, class_colors_plate = darknet.load_network(license_config,data_license,license_weights,batch_size=1)
    network_number, class_names_number,class_colors_number = darknet.load_network(number_config,data_number,number_weights,batch_size=1)

    while True:
        
        image_name = input("Enter Image Path: ")
        input_type = check_input_type(image_name)
        if input_type == 'video':
            print("Wrong input type, try again")
            continue
        
        image_s = cv2.imread(image_name)

        cropped_images=[]
        bcords=[]
        boxes = [] 
        results = []

        detections = image_detection(image_s, network_plate, class_names_plate, class_colors_plate, thresh)
         
        width, height = image_s.shape[1], image_s.shape[0]
        scaled_width = width/network_size
        scaled_height = height/network_size
         
        if len(detections)>0:

         
                for detection in detections:
                    coordinates = list(detection[2])  # Pobranie trzeciego elementu (krotek) z danej krotki
                    boxes.append(coordinates)  # Dodanie koordynatów do nowej tablicy
       

                for box in boxes:
                    xmin, ymin, xmax, ymax =darknet.bbox2points(box)
                    minbcords = [xmin*scaled_width,ymin*scaled_height,xmax*scaled_width,ymax*scaled_height]
                    bcords.append(minbcords)


                for cord in bcords:
                    cropped_image = image_s[int(cord[1])-5:int(cord[3])+5,int(cord[0])-5:int(cord[2])+5]
                    cropped_images.append(cropped_image)

                for crop in cropped_images:
                    recognition = image_detection(crop,network_number,class_names_number,class_colors_number,thresh)
                    result, avg_score = read_license_plate(crop.shape,recognition)
                    results.append(result) 
        
                img= Image.fromarray(image_s,'RGB')
                im1=ImageDraw.Draw(img)
                
                for idx, bcord in enumerate(bcords):
                    im1.text((bcord[0],bcord[3]+3),results[idx],font=font,fill=(255,255,255),stroke_width=3,stroke_fill=(0,0,0))
                    im1.text((bcord[0],bcord[1]-30),"license_plate "+"{:.2f}".format(avg_score)+'%',font=font,stroke_width=3,stroke_fill=(0,0,0))
            
                imp=np.array(img)

                for idx, bcord in enumerate(bcords):
                    imp = cv2.rectangle(imp,(int(bcord[0]),int(bcord[1])),(int(bcord[2]),int(bcord[3])),(255,0,0),3)

        plt.figure()
        plt.imshow(cv2.cvtColor(imp,cv2.COLOR_BGR2RGB))
        plt.show()    
        index += 1


if __name__ == "__main__":

    main()
