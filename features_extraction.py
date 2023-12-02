#------------------------------------------------------------------#
# extract features from an image using the pretrained model vgg16  #
#------------------------------------------------------------------#

#--------------------------------------------------------------------# 
# essential library: PIL.Image, torchvision                          #
# step by step:                                                      #
#   - select device                                                  #
#   - load an image (numpyndarray) and convert to tensor.            #
#   - permute [height, width, channel] to [channel, height, width].  #
#   - add batch_size to dim=0.                                       #
#   - apply preprocessing of vgg16 pretrained model to our input.    #
#   - load weights and pretrained model                              #
#--------------------------------------------------------------------#


from PIL import Image
import torchvision
import torch 
from core.image_utils import image_from_url

# check cuda if it's available
device = "cuda" if torch.cuda.is_available() else "cpu"

# select url and load an image from this url
url = "http://farm1.staticflickr.com/133/330657765_4c19d29015_z.jpg"
img = image_from_url(url=url)

# convert image numpydarray to tensor
img_tensor = torch.from_numpy(img)
# permute from [height, width, channel] to [channel, heighht, width]
img_tensor = img_tensor.permute(dims=[2, 0, 1]) # torch.Size([3, 473, 640])

# add batch_size
img_tensor = img_tensor.unsqueeze(dim=0) # torch.Size([1, 3, 473, 640])
img_tensor = img_tensor.to(device)

# load weights and pretrained model
weights_vgg16 = torchvision.models.VGG16_Weights.DEFAULT
model_vgg16 = torchvision.models.vgg16(weights=weights_vgg16)

# if cuda is available, move params of model to it
model_vgg16 = model_vgg16.to(device)

transforms_vgg16 = weights_vgg16.transforms()
img_tensor = transforms_vgg16(img_tensor)

# compute output
#---------------------------------------------------------------------------------------------------#
# the FC7 layer of VGG16 extracts a 4096-dimensional vector representation of the input image.      #
# This vector representation contains high-level information about the image, such as the objects   #
# that are present in the image, the relationships between the objects, and the overall scene.      #
#---------------------------------------------------------------------------------------------------#
output = model_vgg16.classifier[0:6](model_vgg16.avgpool(model_vgg16.features(img_tensor)).reshape(1, -1)) # torch.Size([1, 4096])
print(output.shape)