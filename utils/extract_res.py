import h5py
import torch
from torchvision import transforms
from PIL import Image
scenes = ["FloorPlan_Train1_1"]
data_dir = "dump"
method = "res18"

class ScaleBothSides(object):
    """Rescales the input PIL.Image to the given 'size'.
    'size' will be the size of both edges, and this can change aspect ratio.
    size: output size of both edges
    interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        return img.resize((self.size, self.size), self.interpolation)


def resnet_input_transform(input_image, im_size):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    all_transforms = transforms.Compose([
        transforms.ToPILImage(),
        ScaleBothSides(im_size),
        transforms.ToTensor(),
        normalize,
    ])
    transformed_image = all_transforms(input_image)
    return transformed_image

for scene in scenes:
    images = h5py.File('{}/{}/images.hdf5'.format(data_dir, scene), 'r')
    features = h5py.File('{}/{}/{}.hdf5'.format(data_dir, scene, method), 'w')

    for k in images:
        frame = resnet_input_transform(images[k][:], 224)
        frame = torch.Tensor(frame)
        if torch.cuda.is_available():
            frame = frame.cuda()
        frame = frame.unsqueeze(0)

        v = model(frame)
        v = v.view(512, 7, 7)

        v = v.cpu().numpy()
        features.create_dataset(k, data=v)

    images.close()
    features.close()