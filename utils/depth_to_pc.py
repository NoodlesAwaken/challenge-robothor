from ai2thor.controller import Controller
import numpy as np
import cv2

def intrinsic_from_fov(height=480, width=640, fov=79):
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

K = intrinsic_from_fov(480, 640)
print(K)
K_inv = np.linalg.inv(K)

controller = Controller(
    agentMode="locobot",
    visibilityDistance=1,
    scene="FloorPlan_Train1_5",
    gridSize=0.25,
    rotateStepDegrees=90,
    renderDepthImage=True,
    renderInstanceSegmentation=False,
    width=640,
    height=480,
    fieldOfView=63.453048374758716
)

positions = controller.step(
    action="GetReachablePositions"
).metadata["actionReturn"]

# import random
# random.seed(19)
# position = random.choice(positions)
# controller.step(
#     action="Teleport",
#     position=positions[9],
# )
# print(len(positions))

controller.step(
    action="Teleport",
    position=positions[28],
)

controller.step(
    action="Teleport",
    position=positions[212],
)

from PIL import Image
import matplotlib.pyplot as plt
# plt.imshow(Image.fromarray(controller.last_event.frame))
# plt.show()

event = controller.step(
    action="RotateLeft",
    degrees=79
)

# depth = Image.fromarray(event.depth_frame[::4, ::4]*100)
# depth = event.depth_frame[::4, ::4]
# depth = depth.repeat(4, axis=0).repeat(4, axis=1)
image = Image.fromarray(event.frame[::4, ::4])
depth = event.depth_frame

# Use Image.NEAREST (0), Image.LANCZOS (1), Image.BILINEAR (2), Image.BICUBIC (3), Image.BOX (4) or Image.HAMMING (5)
# depth.save('depth.tiff') 
print(event.metadata['agent'])
import open3d as o3d
print(event.depth_frame.shape)
d = o3d.geometry.Image(depth.astype(np.float32))

def pixel_coord_np(width, height):
    """
    Pixel in homogenous coordinate
    Returns:
        Pixel coordinate:       [3, width * height]
    """
    x = np.linspace(0, width - 1, width).astype(np.int)
    y = np.linspace(0, height - 1, height).astype(np.int)
    [x, y] = np.meshgrid(x, y)
    return np.vstack((x.flatten(), y.flatten(), np.ones_like(x.flatten())))

pixel_coords = pixel_coord_np(640, 480)  # [3, npoints]

# Apply back-projection: K_inv @ pixels * depth
cam_coords = K_inv[:3, :3] @ pixel_coords * depth.flatten()
print("K inv: \n{}".format(K_inv))
print("K inv[:3, :3]: \n{}".format(K_inv[:3, :3]))


# Limit points to 150m in the z-direction for visualisation
cam_coords = cam_coords[:, np.where(cam_coords[2] <= 150)[0]]
print('cam_coords: {}'.format(cam_coords.shape))

all_coords = []
all_coords.append(cam_coords)

# rotate 30 degrees left: actural -30
# [ cos(a)  0   sin(a)
#   0       1   0
#   -sin(a) 0   cos(a)]
# R = np.matrix('0.866 0 -0.5; 0 1 0; 0.5 0 0.866')

for i in range(11):

    # print("rotate {}".format(i+1))
    event = controller.step(
        action="RotateLeft",
        degrees=30
    )

    # depth = event.depth_frame[::4, ::4]
    # depth = depth.repeat(4, axis=0).repeat(4, axis=1)

    depth = event.depth_frame
    cam_coords = K_inv[:3, :3] @ pixel_coords * depth.flatten()

    degrees = -30 * (i + 1)
    R = np.matrix([
        [np.cos(np.deg2rad(degrees)), 0, np.sin(np.deg2rad(degrees))],
        [0, 1, 0],
        [-np.sin(np.deg2rad(degrees)), 0, np.cos(np.deg2rad(degrees))]
    ])

    cam_coords = cam_coords[:, np.where(cam_coords[2] <= 150)[0]]

    # np.matmul would change array into matrix thus cause problem
    rot_coords = np.asarray(np.matmul(R, cam_coords))
    all_coords.append(rot_coords)

# Visualize
pcd_cam = o3d.geometry.PointCloud()
pcd_cam.points = o3d.utility.Vector3dVector(cam_coords.T[:, :3])
points = np.asarray(pcd_cam.points)

# x_max = np.max(points[:, 0])
# x_min = np.min(points[:, 0])
# y_max = np.max(points[:, 1])
# y_min = np.min(points[:, 1])
# z_max = np.max(points[:, 2])
# z_min = np.min(points[:, 2])
# print(x_max)
# print(x_min)
# print(y_max)
# print(y_min)
# print(z_max)
# print(z_min)

# Flip it, otherwise the pointcloud will be upside down
pcd_cam.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
o3d.visualization.draw_geometries([pcd_cam])
pcd_cam.transform([[0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1]])
# o3d.visualization.draw_geometries([pcd_cam])

def project_topview(cam_points):
    """
    Draw the topview projection
    """
    max_longitudinal = 10
    window_x = (-10, 10)
    window_y = (-10, max_longitudinal)        
    
    fig, axes = plt.subplots(figsize=(12, 12))
    axes.set_xlim(window_x)
    axes.set_ylim(window_y)
    for pts in cam_points:
        x, y, z = pts
        # flip the y-axis to positive upwards
        y = - y

        # We sample points for points less than 70m ahead and above ground
        # Camera is mounted 1m above on an ego vehicle
        ind = np.where((z < max_longitudinal) & (y > -0.8) & (y < 1.6))
        bird_eye = pts[:3, ind]

        # Color by radial distance
        # dists = np.sqrt(np.sum(bird_eye[0:2:2, :] ** 2, axis=0))
        # axes_limit = 10
        # colors = np.minimum(1, dists / axes_limit / np.sqrt(2))

        # Draw Points
        axes.scatter(bird_eye[0, :], bird_eye[2, :], s=0.1, c='#000000')

    plt.gca().set_aspect('equal')
    plt.show()

# Do top view projection
project_topview(all_coords)