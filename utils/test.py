lists = [0, 1, 2, 3]

list = lists[0:2] + lists[3:4]
print(list)


x = 30

if x == 30 or 60:
    print("true")

# import os
# import sys
# sys.path.append(os.getcwd())
import h5py
filename = "dump/FloorPlan_Train1_1/depth.hdf5"

# print(os.getcwd())

# with h5py.File(filename, "r") as f:
    # List all groups
    # print("Keys: %s" % f.keys())
    # a_group_key = list(f.keys())[0]

    # # Get the data
    # data = list(f[a_group_key])

import numpy as np
def intrinsic_from_fov(height, width, fov=79):
    """
    Basic Pinhole Camera Model
    intrinsic params from fov and sensor width and height in pixels
    Returns:
        K:      [4, 4]
    """
    px, py = (width / 2, height / 2)
    hfov = fov / 360. * 2. * np.pi
    fx = width / (2. * np.tan(hfov / 2.))
    vfov = 2. * np.arctan(np.tan(hfov / 2) * height / width)
    fy = height / (2. * np.tan(vfov / 2.))

    return np.array([[fx, 0, px, 0.],
                     [0, fy, py, 0.],
                     [0, 0, 1., 0.],
                     [0., 0., 0., 1.]])


matrix = intrinsic_from_fov(480, 640)
print(matrix)
# 63.453048374758716

from ai2thor.controller import Controller

# controller = Controller(
#     agentMode="locobot",
#     visibilityDistance=15,
#     width=640,
#     height=480,
#     fieldOfView=63.453048374758716,
#     gridSize=0.25,
#     rotateStepDegrees=75,
#     scene="FloorPlan_Train1_1"
#     # scene="FloorPlan_test-challenge1_1"
# )

# count the number of starting postions that can see the target in a 360 view

# import json
# import gzip

# jsonfilename = "data/test/episodes/FloorPlan_test-challenge{}_{}.json.gz".format(1, 1)
# with gzip.open(jsonfilename, 'r') as fin:
#     data = json.loads(fin.read().decode('utf-8'))
#     print(data)

# controller.reset(scene='FloorPlan_test-challenge1_1')

# vis_count = 0
# for a in range(12):
#     for b in range(5):

#         jsonfilename = "data/train/episodes/FloorPlan_Train{}_{}.json.gz".format(a+1, b+1)
#         with gzip.open(jsonfilename, 'r') as fin:
#             data = json.loads(fin.read().decode('utf-8'))

#         print(data[0]['id'])
#         for episode in data:
#             # print("id: " + episode['id'])
#             # print("target: " + episode['object_type'])
#             controller.step(
#                 action='Teleport',
#                 position=episode['initial_position'],
#                 rotation=episode['initial_orientation']
#             )
#             objs = controller.last_event.metadata['objects']

#             visibility = False
#             for obj in objs:
#                 if obj['objectType'] == episode['object_type'] and obj['visible'] == True and obj['obstructed'] == False:
#                     visibility = True

            
#             for i in range(4):
#                 controller.step(
#                     action='RotateLeft',
#                     degrees=75
#                 )

#                 objs = controller.last_event.metadata['objects']
#                 for obj in objs:
#                     if obj['objectType'] == episode['object_type'] and obj['visible'] == True and obj['obstructed'] == False:
#                         visibility = True
            
#             if visibility:
#                 vis_count += 1
#         print('seen: {}'.format(vis_count))

# print(vis_count)
# print(vis_count/108000)


from offline_controller import OfflineControllerWithSmallRotation

controller = OfflineControllerWithSmallRotation()
# controller.start()
controller.reset(scene_name="FloorPlan_Train11_3")
controller.mapping()
# c = Controller(
#     agentMode='locobot',
#     scene='FloorPlan_Train5_1',
#     fieldOfView=63.453048374758716,
#     width=640,
#     height=480,
#     gridSize=0.25
# )
# c.reset(scene="FloorPlan_Train5_2")
# c.step(action="RotateLeft", degrees=90)
# c.step(action="RotateLeft", degrees=90)
# c.step(
#     action="RotateLeft",
#     degrees=30
# )
# controller.step(dict(action="Initialize", fieldOfView=63.453048374758716))
import matplotlib.pyplot as plt

# import time
# start = time.time()
# for i in range(1000):
#     controller.step(dict(action="RotateLeft"))
#     img = controller.get_image()
# end = time.time()
# print("dump data 1000 steps: {} seconds".format(end - start))
plt.imshow(controller.get_image())
plt.show()
controller.step(dict(action="RotateLeft"))
plt.imshow(controller.get_image())
plt.show()
controller.mapping()
controller.step(dict(action="LookUp"))
plt.imshow(controller.get_image())
plt.show()
controller.mapping()
controller.step(dict(action="LookUp"))
plt.imshow(controller.get_image())
plt.show()
controller.mapping()
controller.step(dict(action="LookDown"))
plt.imshow(controller.get_image())
plt.show()
controller.mapping()
controller.step(dict(action="LookDown"))
plt.imshow(controller.get_image())
plt.show()
controller.mapping()
controller.step(dict(action="LookDown"))
plt.imshow(controller.get_image())
plt.show()
controller.mapping()
controller.step(dict(action="MoveAhead"))
plt.imshow(controller.get_image())
plt.show()
controller.mapping()
controller.step(dict(action="RotateLeft"))
plt.imshow(controller.get_image())
plt.show()
controller.mapping()
controller.step(dict(action="MoveAhead"))
plt.imshow(controller.get_image())
plt.show()
controller.mapping()
controller.step(dict(action="RotateLeft"))
plt.imshow(controller.get_image())
plt.show()
controller.mapping()
controller.step(dict(action="MoveAhead"))
plt.imshow(controller.get_image())
plt.show()
controller.mapping()
controller.step(dict(action="MoveAhead"))
plt.imshow(controller.get_image())
plt.show()
controller.mapping()
controller.step(dict(action="MoveAhead"))
plt.imshow(controller.get_image())
plt.show()
controller.mapping()
controller.step(dict(action="RotateLeft"))
plt.imshow(controller.get_image())
plt.show()
controller.mapping()
controller.step(dict(action="RotateLeft"))
plt.imshow(controller.get_image())
plt.show()
controller.mapping()
controller.step(dict(action="MoveAhead"))
plt.imshow(controller.get_image())
plt.show()
controller.mapping()
controller.step(dict(action="RotateLeft"))
plt.imshow(controller.get_image())
plt.show()
controller.mapping()
controller.step(dict(action="MoveAhead"))
plt.imshow(controller.get_image())
plt.show()
controller.mapping()
controller.step(dict(action="RotateLeft"))
plt.imshow(controller.get_image())
plt.show()
controller.mapping()
controller.step(dict(action="MoveAhead"))
plt.imshow(controller.get_image())
plt.show()
controller.mapping()
controller.step(dict(action="MoveAhead"))
plt.imshow(controller.get_image())
plt.show()
controller.mapping()
controller.step(dict(action="MoveAhead"))
plt.imshow(controller.get_image())
plt.show()
controller.mapping()
controller.step(dict(action="MoveAhead"))
plt.imshow(controller.get_image())
plt.show()
controller.mapping()
