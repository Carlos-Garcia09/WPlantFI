import torch
import torchvision
from utils import set_seed, load_model, save, get_model, update_optimizer, get_data
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import json

from torchvision.models import resnet18
import torch.nn.functional as F
import re




custom_image_path = "rosa.jpg"

# def convert_json_with_regex(json_str):
#     # Define a regular expression pattern to capture key-value pairs
#     pattern = r'"(\d+)":\s*"([^"]+)"'

#     # Find all matches in the input JSON string
#     matches = re.findall(pattern, json_str)

#     converted_json = {}
#     for index, (plant_id, plant_name) in enumerate(matches):
#         converted_json[str(index)] = {"plant_id": plant_id, "plant_name": plant_name}

#     return converted_json


with open('order_DataSet.json', 'r') as json_file:
    data = json.load(json_file)
    
# json_data = json.dumps(data)
# converted_json = convert_json_with_regex(json_data)

# output_file_path = 'PlantNet-300K/order_DataSet.json'  # Specify the path of the new file

# with open(output_file_path, 'w') as output_file:
#     json.dump(converted_json, output_file, indent=4)



#my model /results/xp1/xp1_weights_best_acc.tar

filename = 'results/xp1/xp1_weights_best_acc.tar' # pre-trained model path
use_gpu = True  # load weights on the gpu
model = resnet18(num_classes=1081) # 1081 classes in Pl@ntNet-300K

load_model(model, filename, False)

model.eval()

# Open and preprocess the image
image = Image.open(custom_image_path)
preprocess = transforms.Compose([transforms.Resize(256),
                                 transforms.CenterCrop(224),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

input_tensor = preprocess(image)
input_tensor = input_tensor.unsqueeze(0)  # Add a batch dimension

# Pass the preprocessed image through your PyTorch model to obtain the model's output.
with torch.no_grad():
    output = model(input_tensor)
    



probabilities = F.softmax(output, dim=1)  # Apply softmax

_, predicted_class = probabilities.max(1)  # Get the predicted class index


# 'predicted_class' contains the index of the predicted class

plt.figure()
plt.imshow(image)
plt.title('Input Image')
print(f'Predicted Class: {predicted_class.item()}')


if str(predicted_class.item()) in data:
    result = data[str(predicted_class.item())]['plant_name']
    print(f"Predicted Class: {predicted_class.item()}")
    print(f"El nombre de la planta es: {result}")
else:
    print("Predicted class not found in the JSON data.")


plt.show()






