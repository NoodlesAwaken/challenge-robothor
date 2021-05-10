from ai2thor.controller import Controller
import numpy as np
import cv2
import random

objs = ['Cup', 'Newspaper', 'Fork', 'ShelvingUnit', 'Desk', 'ArmChair', 'DeskLamp', 'Apple', 'TVStand', 'CD', 'Pen', 'Television', 'Laptop', 'Pot', 'DiningTable', 'Statue', 'ButterKnife', 'CoffeeTable', 'Vase', 'Drawer', 'GarbageCan', 'Dresser', 'Pencil', 'Bed', 'SideTable', 'Box', 'FloorLamp', 'AlarmClock', 'PepperShaker', 'Sofa', 'Mug', 'Chair', 'Candle', 'SprayBottle', 'Pillow', 'Watch', 'TennisRacket', 'RemoteControl', 'SaltShaker', 'Plate', 'HousePlant', 'Painting', 'Bottle', 'Book', 'Bowl', 'TeddyBear', 'BaseballBat', 'Shelf', 'BasketBall', 'CellPhone']
print("apple: {}".format(objs.index("Apple")))
print("alarmClock: {}".format(objs.index("AlarmClock")))
print("baseballBat: {}".format(objs.index("BaseballBat")))
print("bowl: {}".format(objs.index("Bowl")))
print("garbageCan: {}".format(objs.index("GarbageCan")))
print("housePlant: {}".format(objs.index("HousePlant")))
print("laptop: {}".format(objs.index("Laptop")))
print("sprayBottle: {}".format(objs.index("SprayBottle")))
print("mug: {}".format(objs.index("Mug")))
print("TV: {}".format(objs.index("Television")))
print("Vase: {}".format(objs.index("Vase")))

# print("low appearances")
# print(objs[35])
# print(objs[16])
# print(objs[28])
# print(objs[38])
# print(objs[45])
# print("high appearances")
# print(objs[24])
# print(objs[31])
# print(objs[41])

def plot_one_box(x, im, color=None, label=None, line_thickness=3):
    # Plots one bounding box on image 'im' using OpenCV
    assert im.data.contiguous, 'Image not contiguous. Apply np.ascontiguousarray(im) to plot_on_box() input image.'
    tl = line_thickness or round(0.002 * (im.shape[0] + im.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(im, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(im, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(im, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

def get_bbox(bbox):
    x0 = (bbox[2] + bbox[0]) / 1280
    y0 = (bbox[3] + bbox[1]) / 960
    w = (bbox[2] - bbox[0]) / 640
    h = (bbox[3] - bbox[1]) / 480
    return [x0, y0, w, h]

controller = Controller(
    agentMode="locobot",
    visibilityDistance=1,
    scene="FloorPlan_Val2_5",
    # scene="FloorPlan_Train12_3",
    gridSize=0.25,
    rotateStepDegrees=90,
    renderDepthImage=True,
    renderInstanceSegmentation=True,
    width=640,
    height=480,
    fieldOfView=63.453048374758716
)

event = controller.step(
    action='RotateLeft',
    degrees=30
)

image = event.frame
depth = event.depth_frame
segmentation = event.instance_segmentation_frame
mask = event.instance_masks
label = event.instance_detections2D
meta = event.metadata


# print("==========================")
print(label)
print(len(label))
from PIL import Image
import matplotlib.pyplot as plt

# plt.imshow(Image.fromarray(image))
# plt.show()
# plt.imshow(Image.fromarray(depth * 100))
# plt.show()
# plt.imshow(Image.fromarray(segmentation))
# plt.show()
# print(label)
img = Image.fromarray(image)
img.save("original.png")
# result = Image.fromarray(image)
image = np.ascontiguousarray(image)

num_box = 0
lines = []
for obj in meta["objects"]:
    if obj["objectId"] in label:
        num_box += 1
        box = get_bbox(label[obj["objectId"]])
        bbox = str("{} {} {} {} {}\n".format(objs.index(obj["objectType"]), box[0], box[1], box[2], box[3]))
        print(bbox)
        print(obj["objectType"])
        lines.append(bbox)
        plot_one_box(label[obj["objectId"]], image, label=obj["objectType"], line_thickness=2)

print(lines)
print("number of box: {}".format(num_box))
if num_box > 5:
    f = open("result.txt", "a")
    f.writelines(lines)
    f.close()
result = Image.fromarray(image)
plt.imshow(result)
plt.show()
result.save("gt.png")

# from ai2thor.controller import Controller
# controller = Controller(scene='FloorPlan1')

# for m in controller.last_event.metadata:
#     print(m)

# objs = set()
# for i in range(12):
#     for j in range(5):
#         controller.reset(scene="FloorPlan_Train{}_{}".format(i+1, j+1))

#         for obj in controller.last_event.metadata["objects"]:
#             # print(obj["objectType"])
#             objs.add(obj["objectType"])

# for i in range(3):
#     for j in range(5):
#         controller.reset(scene="FloorPlan_Val{}_{}".format(i+1, j+1))

#         for obj in controller.last_event.metadata["objects"]:
#             # print(obj["objectType"])
#             objs.add(obj["objectType"])



 
# print("==========================")
# print(objs)
# print(len(objs))

# for m in controller.last_event.metadata["objects"]:
#     print(m["name"])

# print("==========================")

# # for m in controller.last_event.metadata["objects"]:
#     # print(m)
# print(meta["objects"][0])

