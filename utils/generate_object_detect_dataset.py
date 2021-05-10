import gzip
import json
import keyboard
import time
import matplotlib.pyplot as plt
import numpy as np

from ai2thor.controller import Controller
from PIL import Image
objs = [
    'Cup', 'Newspaper', 'Fork', 'ShelvingUnit', 'Desk', 'ArmChair', 'DeskLamp', 'Apple', 'TVStand', 'CD', 'Pen', 'Television', 
    'Laptop', 'Pot', 'DiningTable', 'Statue', 'ButterKnife', 'CoffeeTable', 'Vase', 'Drawer', 'GarbageCan', 'Dresser', 'Pencil', 
    'Bed', 'SideTable', 'Box', 'FloorLamp', 'AlarmClock', 'PepperShaker', 'Sofa', 'Mug', 'Chair', 'Candle', 'SprayBottle', 
    'Pillow', 'Watch', 'TennisRacket', 'RemoteControl', 'SaltShaker', 'Plate', 'HousePlant', 'Painting', 'Bottle', 'Book', 
    'Bowl', 'TeddyBear', 'BaseballBat', 'Shelf', 'BasketBall', 'CellPhone']

controller = Controller(
    agentType="stochastic",
    agentMode="locobot",
    continuousMode=True,
    applyActionNoise=False,
    visibilityDistance=1,
    # headless=True,
    scene="FloorPlan_Train1_1",
    gridSize=0.25,
    snapToGrid=False,
    # movementGaussianSigma=0.0,
    rotateStepDegrees=30,
    # rotateGaussianSigma=0.0,
    renderDepthImage=False,
    renderInstanceSegmentation=True,
    width=640,
    height=480,
    fieldOfView=63.453048374758716
)

def get_bbox(bbox):
    x0 = (bbox[2] + bbox[0]) / 1280
    y0 = (bbox[3] + bbox[1]) / 960
    w = (bbox[2] - bbox[0]) / 640
    h = (bbox[3] - bbox[1]) / 480
    return [x0, y0, w, h]

# p = controller.last_event.metadata["agent"]["position"]
# # r = controller.last_event.metadata["agent"]["rotation"]
r = {'x': 0.0, 'y': 0.0, 'z': 0.0} 
# h = controller.last_event.metadata["agent"]["cameraHorizon"]
# print(p)
# print(r) 
# # print(h)
# ps = 0
# for i in range(12):
#     for j in range(5):
#         controller.reset(scene="FloorPlan_Train{}_{}".format(i+1, j+1))
#         positions = controller.step(action="GetReachablePositions").metadata["actionReturn"]
#         ps = ps + len(positions)
#         print(ps)


############### 17501 positions in total ###############

for i in range(12):
    count = 0
    for j in range(5):
        controller.reset(scene="FloorPlan_Train{}_{}".format(i+1, j+1))
        positions = controller.step(action="GetReachablePositions").metadata["actionReturn"]
        for p in positions:
            controller.step("Teleport", position=p, rotation=r)
            for k in range(4):
                event = controller.step("RotateRight", degrees=90*k)
                frame = Image.fromarray(event.frame)
                # seg = Image.fromarray(event.instance_segmentation_frame)
                # depth = event.depth_frame
                # segmentation = event.instance_segmentation_frame
                # mask = event.instance_masks
                label = event.instance_detections2D
                meta = event.metadata

                num_box = 0
                lines = []
                for obj in meta["objects"]:
                    if obj["objectId"] in label:
                        num_box += 1
                        box = get_bbox(label[obj["objectId"]])
                        bbox = str("{} {} {} {} {}\n".format(objs.index(obj["objectType"]), box[0], box[1], box[2], box[3]))
                        lines.append(bbox)
                        # plot_one_box(label[obj["objectId"]], image, label=obj["objectType"], line_thickness=2)

                if num_box > 5:
                    count += 1
                    f = open("object_detection_dataset/FloorPlan_Train{}_{}_frame{}.txt".format(i+1, j+1, count), "a")
                    f.writelines(lines)
                    f.close()
                    frame.save('object_detection_dataset/FloorPlan_Train{}_{}_frame{}.png'.format(i+1, j+1, count))
                # seg.save('Train{}/seg/FloorPlan_Train{}_{}_seg{}.png'.format(i+1, i+1, j+1, count))


# for i in range(12):
#     # time.sleep(2)
#     # img = controller.last_event.instance_segmentation_frame
#     # plt.imshow(img)
#     # plt.show()
#     controller.step(
#         action="TeleportFull",
#         position=p,
#         rotation=r,
#         horizon=h
#     )

#     # plt.imshow(img)
#     # plt.show()
#     event = controller.step(action="RotateLeft", degrees=30*i)
#     print("before action {}".format(i))
#     print(controller.last_event.metadata["agent"])
#     # time.sleep(1)
#     frame = Image.fromarray(event.frame)
#     seg = Image.fromarray(event.instance_segmentation_frame)
#     frame.save('frame{}.png'.format(i))
#     seg.save('seg{}.png'.format(i))
#     controller.step(action="MoveAhead")
#     print("after action {}".format(i))
#     print(controller.last_event.metadata["agent"])
    # time.sleep(1)
    # controller.step("MoveBack")
    # print(controller.last_event.metadata["agent"])
    # time.sleep(1)


#     # time.sleep(2)


# positions = controller.step(
#     action="GetReachablePositions"
# ).metadata["actionReturn"]

# print(positions)

# print("====================")



# with gzip.open('data/train/episodes/FloorPlan_Train1_1.json.gz', 'rb') as f:
#     file_content = f.read()


# js = json.loads(file_content)
# for j in js:
#     print(j)


# plt.imshow(Image.fromarray(image))
# plt.show()
# plt.imshow(Image.fromarray(depth * 100))
# plt.show()
# plt.imshow(Image.fromarray(segmentation))
# plt.show()
# print(label)

# result = Image.fromarray(image)