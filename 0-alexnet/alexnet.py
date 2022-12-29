import json
import torch
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
import torchvision.models as models
from PIL import Image
from alexnet_pytorch import AlexNet
from vgg_pytorch import VGG 
from resnet_pytorch import ResNet

#from googletrans import Translator


input_image = Image.open("0-alexnet/images/test7.jpg")
input_image = img = input_image.convert('RGB')

plt.figure()
plt.imshow(input_image)
plt.show(block=True)
   
preprocess = transforms.Compose([
  transforms.Resize(256),
  transforms.CenterCrop(224),
  transforms.ToTensor(),
  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

input_image = preprocess(input_image)
input_image = torch.FloatTensor(input_image)
input_batch = input_image.unsqueeze(0)  # create a mini-batch as expected by the model

print(input_batch)

# Load class names
labels_map = json.load(open("0-alexnet/labels_map.txt"))
labels_map = [labels_map[str(i)] for i in range(1000)]
 
# Classify with AlexNet
model_1 = AlexNet.from_pretrained("alexnet")
model_2 = VGG.from_pretrained('vgg16')
model_3 = ResNet.from_pretrained('resnet18', num_classes=10)
model_4 = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True)

model_1.eval()
model_2.eval()
model_3.eval()
model_4.eval()

# move the input and model to GPU for speed if available
if torch.cuda.is_available():
  input_batch = input_batch.to("cuda")
  model_1.to("cuda")
  model_2.to("cuda")
  model_3.to("cuda")
  model_4.to("cuda")
  
with torch.no_grad():
    output_1 = model_1(input_batch)
    output_2 = model_2(input_batch)
    output_3 = model_3(input_batch)
    output_4 = model_4(input_batch)

with torch.no_grad():
  logits_1 = model_1(input_batch)
  logits_2 = model_2(input_batch)
  logits_3 = model_3(input_batch)
  logits_4 = model_4(input_batch)
preds_1 = torch.topk(logits_1, k=5).indices.squeeze(0).tolist()
preds_2 = torch.topk(logits_2, k=5).indices.squeeze(0).tolist()
preds_3 = torch.topk(logits_3, k=5).indices.squeeze(0).tolist()
preds_4 = torch.topk(logits_4, k=5).indices.squeeze(0).tolist()

print("------------------------------------------alexnet")

outfile = open('0-alexnet/output.txt', 'w')  

for idx in preds_1:
  label = labels_map[idx]
  prob = torch.softmax(logits_1, dim=1)[0, idx].item()
  print(f"{label:<75} ({prob * 100:.2f}%)")
  
  outfile.write(f"{label:<75} ({prob * 100:.2f}%)"+"\n") 
  outfile.write(str(idx))
  outfile.write('\n')  
print("------------------------------------------vgg16")

for idx in preds_2:
  label = labels_map[idx]
  prob = torch.softmax(logits_2, dim=1)[0, idx].item()
  print(f"{label:<75} ({prob * 100:.2f}%)")
  
  outfile.write(f"{label:<75} ({prob * 100:.2f}%)"+"\n") 
  outfile.write(str(idx))
  outfile.write('\n')  
print("------------------------------------------resnet18")  
for idx in preds_3:
  label = labels_map[idx]
  prob = torch.softmax(logits_3, dim=1)[0, idx].item()
  print(f"{label:<75} ({prob * 100:.2f}%)")
   
  outfile.write(f"{label:<75} ({prob * 100:.2f}%)"+"\n") 
  outfile.write(str(idx))
  outfile.write('\n')  

print("------------------------------------------InceptionV3")  
for idx in preds_4:
  label = labels_map[idx]
  prob = torch.softmax(logits_4, dim=1)[0, idx].item()
  print(f"{label:<75} ({prob * 100:.2f}%)")
   
  outfile.write(f"{label:<75} ({prob * 100:.2f}%)"+"\n") 
  outfile.write(str(idx))
  outfile.write('\n')  

print("------------------------------------------")
# translator = Translator()
# f = open('0-alexnet/output.txt','r')
# content = f . read()
# print(content)
# translation = translator.translate(content, dest= 'en')
# print(translation.text)
#     # translator= Translator(to_lang="Persian")
#     # translation = translator.translate(outfile)
#     # print (translation)
# outfile = open('0-alexnet/output2.txt', 'w') 
# outfile.write(translation.text) 