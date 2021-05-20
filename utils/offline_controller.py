import importlib
from collections import deque
# import json
import ujson as json
import copy
import time
import random
import os
import platform
import numpy as np
import torch

from ai2thor.controller import Controller, distance
from base_controller import BaseController
from bfs_controller import ExhaustiveBFSController, ThorAgentState
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def project_topview(cam_points):
    # start_plt = time.time()
    x, y, z = cam_points
    # flip the y-axis to positive upwards
    y = - y

    # We sample points for points above ground and under ceiling
    ind = (y > -1.2) & (y < 0.9)
    bird_eye = cam_points[:3, ind]
    bird_eye = bird_eye[[0, 2], :]

    m = torch.zeros(500, 500).cuda()
    index = ((bird_eye + 10) / 0.04).long()

    m[index[0, :], index[1, :]] = 1

    # end_plt = time.time()
    # print("bin time: {}".format(end_plt - start_plt))

    # m = m.cpu()
    # plt.imshow(m)
    # plt.show()

    # bird_eye = bird_eye.cpu()
    # window_x = (-10, 10)
    # window_y = (-10, 10)   
    # fig, axes = plt.subplots(figsize=(12, 12))
    # axes.set_xlim(window_x)
    # axes.set_ylim(window_y)
    # axes.scatter(bird_eye[0, :], bird_eye[1, :], s=0.1, c="#000000")
    # plt.show()
    return m
    


def get_projection_matrix(height=480, width=640, fov=79):
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

    K = np.array([[fx, 0, px, 0.],
                     [0, fy, py, 0.],
                     [0, 0, 1., 0.],
                     [0., 0., 0., 1.]])
    K_inv = np.linalg.inv(K)

    x = np.linspace(0, width - 1, width).astype(np.int)
    y = np.linspace(0, height - 1, height).astype(np.int)
    [x, y] = np.meshgrid(x, y)

    return torch.from_numpy(K_inv[:3, :3] @ np.vstack((x.flatten(), y.flatten(), np.ones_like(x.flatten())))).cuda()



def get_rotate_matrix():
    rotate_matrix = []
    for i in range(8):
        # degrees = i * 30
        degrees = i * 45
        R = torch.tensor([
            [np.cos(np.deg2rad(degrees)), 0, np.sin(np.deg2rad(degrees))],
            [0, 1, 0],
            [-np.sin(np.deg2rad(degrees)), 0, np.cos(np.deg2rad(degrees))]
            ], device=device)
 
        rotate_matrix.append(R)

    return rotate_matrix

projection_matrix = get_projection_matrix()
rotate_matrix = get_rotate_matrix()


class OfflineControllerWithSmallRotationEvent:
    """ A stripped down version of an event. Only contains lastActionSuccess, sceneName,
        and optionally state and frame. Does not contain the rest of the metadata. """

    def __init__(self, last_action_success, scene_name, state=None, frame=None):
        self.metadata = {
            "lastActionSuccess": last_action_success,
            "sceneName": scene_name,
        }
        if state is not None:
            self.metadata["agent"] = {}
            self.metadata["agent"]["position"] = state.position()
            self.metadata["agent"]["rotation"] = {
                "x": 0.0,
                "y": state.rotation,
                "z": 0.0,
            }
            self.metadata["agent"]["cameraHorizon"] = state.horizon
        self.frame = frame


class OfflineControllerWithSmallRotation(BaseController):
    """ A stripped down version of the controller for non-interactive settings.
        Only allows for a few given actions. Note that you must use the
        ExhaustiveBFSController to first generate the data used by OfflineControllerWithSmallRotation.
        Data is stored in offline_data_dir/<scene_name>/.
        Can swap the metadata.json for a visible_object_map.json. A script for generating
        this is coming soon. If the swap is made then the OfflineControllerWithSmallRotation is faster and
        self.using_raw_metadata will be set to false.
        Additionally, images.hdf5 may be swapped out with ResNet features or anything
        that you want to be returned for event.frame. """

    def __init__(
        self,
        grid_size=0.25,
        fov=63.453048374758716,
        offline_data_dir="/media/gregory/data/dump/",
        grid_file_name="grid.json",
        graph_file_name="graph.json",
        metadata_file_name='metadata.json',
        images_file_name="images.hdf5",
        depth_file_name="depth.hdf5",
        debug_mode=True,
        actions=["MoveAhead", "RotateLeft", "RotateRight", "LookUp", "LookDown"],
        visualize=False,
        local_executable_path=None
    ):

        super(OfflineControllerWithSmallRotation, self).__init__()
        self.grid_size = grid_size
        self.offline_data_dir = offline_data_dir
        self.grid_file_name = grid_file_name
        self.graph_file_name = graph_file_name
        self.metadata_file_name = metadata_file_name
        self.images_file_name = images_file_name
        self.depth_file_name = depth_file_name
        self.map = None
        self.grid = None
        self.graph = None
        self.metadata = None
        self.images = None
        self.depth = None
        self.controller = None
        self.using_raw_metadata = True
        self.update_map = False
        self.actions = actions
        # Allowed rotations.
        # self.rotations = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
        self.rotations = [0, 45, 90, 135, 180, 225, 270, 315]
        # Allowed horizons.
        self.horizons = [-30, 0, 30]
        self.debug_mode = debug_mode
        self.fov = fov

        self.local_executable_path = local_executable_path

        self.y = None

        self.last_event = None

        # self.controller = ExhaustiveBFSController()
        if self.local_executable_path is not None:
            self.controller.local_executable_path = self.local_executable_path

        self.visualize = visualize

        self.scene_name = None
        self.state = None
        self.init_state = None
        self.last_action_success = True

        self.h5py = importlib.import_module("h5py")
        self.nx = importlib.import_module("networkx")
        self.json_graph_loader = importlib.import_module("networkx.readwrite")

    def start(self):
        if self.visualize:
            self.controller.start()
            self.controller.step(
                dict(action="Initialize", gridSize=self.grid_size, fieldOfView=self.fov)
            )

    def set_init_state(self, state):
        self.init_state = state

    def get_full_state(self, x, y, z, rotation=0.0, horizon=0.0):
        return ThorAgentState(x, y, z, rotation, horizon)

    def get_state_from_str(self, x, z, rotation=0.0, horizon=0.0):
        return ThorAgentState(x, self.y, z, rotation, horizon)

    def reset(self, scene_name=None):

        if scene_name is None:
            scene_name = "FloorPlan28"

        if scene_name != self.scene_name:
            self.scene_name = scene_name
            with open(
                os.path.join(
                    self.offline_data_dir, self.scene_name, self.grid_file_name
                ),
                "r",
            ) as f:
                self.grid = json.load(f)
            with open(
                os.path.join(
                    self.offline_data_dir, self.scene_name, self.graph_file_name
                ),
                "r",
            ) as f:
                graph_json = json.load(f)
            self.graph = self.json_graph_loader.node_link_graph(
                graph_json
            ).to_directed()
            # # meta data loading costs 12 seconds
            with open(
                os.path.join(
                    self.offline_data_dir, self.scene_name, self.metadata_file_name
                ),
                "r",
            ) as f:
                self.metadata = json.load(f)
                # Determine if using the raw metadata, which is structured as a dictionary of
                # state -> metatdata. The alternative is a map of obj -> states where object is visible.
                key = next(iter(self.metadata.keys()))
                try:
                    float(key.split("|")[0])
                    self.using_raw_metadata = True
                except ValueError:
                    self.using_raw_metadata = False
            if self.images is not None:
                self.images.close()
            self.images = self.h5py.File(
                os.path.join(
                    self.offline_data_dir, self.scene_name, self.images_file_name
                ),
                "r",
            )
            if self.depth is not None:
                self.depth.close()
            self.depth = self.h5py.File(
                os.path.join(
                    self.offline_data_dir, self.scene_name, self.depth_file_name
                ),
                "r",
            )

        self.state = self.get_full_state(
            **self.grid[0], rotation=random.choice(self.rotations)
        )
        self.y = self.state.y
        self.last_action_success = True
        self.last_event = self._successful_event()

        if self.visualize:
            self.controller.reset(scene_name)
            self.controller.teleport_to_state(self.state)

    def randomize_state(self):
        self.state = self.get_state_from_str(
            *[float(x) for x in random.choice(list(self.images.keys())).split("|")]
        )
        self.state.horizon = 0
        self.last_action_success = True
        self.last_event = self._successful_event()

        if self.visualize:
            self.controller.teleport_to_state(self.state)

    def back_to_start(self, start):
        self.state = start
        if self.visualize:
            self.controller.teleport_to_state(self.state)

    def step(self, action, raise_for_failure=False):

        if "action" not in action or action["action"] not in self.actions:
            if action["action"] == "Initialize":
                if self.visualize:
                    # self.controller.step(action, raise_for_failure)
                    self.controller.step(action="Initialize")
                return
            raise Exception("Unsupported action.")

        action = action["action"]
        self.update_map = False
        # print("executing action: {}".format(action))

        # next_state = self.controller.get_next_state(self.state, action, True)
        next_state = copy.deepcopy(self.state)
        if action == "MoveAhead":
            self.update_map = True
            if next_state.rotation == 0:
                next_state.z += self.grid_size
                shift = (0, -6)
            elif next_state.rotation == 90:
                next_state.x += self.grid_size
                shift = (-6, 0)
            elif next_state.rotation == 180:
                next_state.z -= self.grid_size
                shift = (0, 6)
            elif next_state.rotation == 270:
                next_state.x -= self.grid_size
                shift = (6, 0)
            elif next_state.rotation == 45:
                next_state.z += self.grid_size
                next_state.x += self.grid_size
                shift = (-6, -6)
            elif next_state.rotation == 135:
                next_state.z -= self.grid_size
                next_state.x += self.grid_size
                shift = (-6, 6)
            elif next_state.rotation == 225:
                next_state.z -= self.grid_size
                next_state.x -= self.grid_size
                shift = (6, 6)
            elif next_state.rotation == 315:
                next_state.z += self.grid_size
                next_state.x -= self.grid_size
                shift = (6, -6)
            # elif next_state.rotation == 30 or 60:
            #     next_state.z += self.grid_size
            #     next_state.x += self.grid_size
            # elif next_state.rotation == 120 or 150:
            #     next_state.z -= self.grid_size
            #     next_state.x += self.grid_size
            # elif next_state.rotation == 210 or 240:
            #     next_state.z -= self.grid_size
            #     next_state.x -= self.grid_size
            # elif next_state.rotation == 300 or 330:
            #     next_state.z += self.grid_size
            #     next_state.x -= self.grid_size
            else:
                raise Exception("Unknown Rotation")
        elif action == "RotateRight":
            # next_state.rotation = (next_state.rotation + 30) % 360
            next_state.rotation = (next_state.rotation + 45) % 360
        elif action == "RotateLeft":
            # next_state.rotation = (next_state.rotation - 30) % 360
            next_state.rotation = (next_state.rotation + 45) % 360
        elif action == "LookUp":
            if next_state.horizon > -30:
                next_state.horizon = next_state.horizon - 30
        elif action == "LookDown":
            if next_state.horizon < 30:
                next_state.horizon = next_state.horizon + 30

        # if self.visualize and next_state is not None:
        #     viz_event = self.controller.step(
        #         dict(
        #             action="Teleport", 
        #             x=next_state.x, 
        #             y=next_state.y, 
        #             z=next_state.z, 
        #             rotation=next_state.rotation, 
        #             horizon=next_state.horizon
        #         )
        #     )
        #     # viz_event = self.controller.step(
        #     #     dict(action="Rotate", rotation=next_state.rotation)
        #     # )
        #     # viz_event = self.controller.step(
        #     #     dict(action="Look", horizon=next_state.horizon)
        #     # )
        #     viz_next_state = self.controller.get_state_from_event(viz_event)
        #     if (
        #         round(viz_next_state.horizon) not in self.horizons
        #         or round(viz_next_state.rotation) not in self.rotations
        #     ):
        #         # return back to original state.
        #         self.controller.teleport_to_state(self.state)

        if next_state is not None:
            next_state_key = str(next_state)
            neighbors = list(self.graph.neighbors(str(self.state)))

            if next_state_key in neighbors:
                self.state = self.get_state_from_str(
                    *[float(x) for x in next_state_key.split("|")]
                )
                self.last_action_success = True
                event = self._successful_event()
                # if self.debug_mode and self.visualize:
                #     if self.controller.get_state_from_event(
                #         viz_event
                #     ) != self.controller.get_state_from_event(event):
                #         print(action)
                #         print(str(self.controller.get_state_from_event(viz_event)))
                #         print(str(self.controller.get_state_from_event(event)))

                #     assert self.controller.get_state_from_event(
                #         viz_event
                #     ) == self.controller.get_state_from_event(event)
                #     assert viz_event.metadata["lastActionSuccess"]

                    # Uncomment if you want to view the frames side by side to
                    # ensure that they are duplicated.
                    # from matplotlib import pyplot as plt
                    # fig = plt.figure()
                    # fig.add_subplot(2,1,1)
                    # plt.imshow(self.get_image())
                    # fig.add_subplot(2,1,2)
                    # plt.imshow(viz_event.frame)
                    # plt.show()
                if self.map is None:
                    self.map = self.mapping()
                elif self.update_map:
                    self.map = self.map.roll(shifts=shift, dims=(0, 1))
                    self.map += self.mapping()

                # plt.imshow(self.map.cpu())
                # plt.show()
                self.last_event = event
                # print("successful")
                return event

        self.last_action_success = False
        self.last_event.metadata["lastActionSuccess"] = False
        # print("failed")
        return self.last_event
    

    def mapping(self):
        if not self.last_action_success:
            return


        coords = str(self.state).split('|')
        start = time.time()

        # start_load = time.time()
        depth_keys = [str("{}|{}|{}|0".format(coords[0], coords[1], rot * 45)) for rot in range(8)]
        depth = [torch.from_numpy(self.get_depth(key)).cuda().flatten() for key in depth_keys]
        # end_load = time.time()

        # start_rotate = time.time()
        cam_coords = [projection_matrix.mul(depth[i]) for i in range(8)]
        all_coords = [torch.mm(rotate_matrix[i], cam_coords[i]) for i in range(8)]
        top = torch.cat(all_coords, 1)
        # end_rotate = time.time()


        # total = time.time() - start
        # print("\ntotal loop time: {}\n".format(total))
        # print("load time: {}".format(end_load - start_load))
        # print("rotate time: {}".format(end_rotate - start_rotate))
        return project_topview(top)



    def shortest_path(self, source_state, target_state):
        return self.nx.shortest_path(self.graph, str(source_state), str(target_state))

    def optimal_plan(self, source_state, path):
        """ This is for debugging. It modifies the state. """
        self.state = source_state
        actions = []
        i = 1
        while i < len(path):
            for a in self.actions:
                next_state = self.controller.get_next_state(self.state, a, True)
                if str(next_state) == path[i]:
                    actions.append(a)
                    i += 1
                    self.state = next_state
                    break

        return actions

    def shortest_path_to_target(self, source_state, objId, get_plan=False):
        """ Many ways to reach objId, which one is best? """
        states_where_visible = []
        if self.using_raw_metadata:
            for s in self.metadata:
                objects = self.metadata[s]["objects"]
                visible_objects = [o["objectId"] for o in objects if o["visible"]]
                if objId in visible_objects:
                    states_where_visible.append(s)
        else:
            states_where_visible = self.metadata[objId]

        # transform from strings into states
        states_where_visible = [
            self.get_state_from_str(*[float(x) for x in str_.split("|")])
            for str_ in states_where_visible
        ]

        best_path = None
        best_path_len = 0
        for t in states_where_visible:
            path = self.shortest_path(source_state, t)
            if len(path) < best_path_len or best_path is None:
                best_path = path
                best_path_len = len(path)
        best_plan = []

        if get_plan:
            best_plan = self.optimal_plan(source_state, best_path)

        return best_path, best_path_len, best_plan

    def visualize_plan(self, source, plan):
        """ Visualize the best path from source to plan. """
        assert self.visualize
        self.controller.teleport_to_state(source)
        time.sleep(0.5)
        for a in plan:
            print(a)
            self.controller.step(dict(action=a))
            time.sleep(0.5)

    def object_is_visible(self, objId):
        if self.using_raw_metadata:
            objects = self.metadata[str(self.state)]["objects"]
            visible_objects = [o["objectId"] for o in objects if o["visible"]]
            return objId in visible_objects
        else:
            return str(self.state) in self.metadata[objId]

    def _successful_event(self):
        return OfflineControllerWithSmallRotationEvent(
            self.last_action_success, self.scene_name, self.state, self.get_image()
        )

    def get_image(self):
        return self.images[str(self.state)][:]

    def get_depth(self, state):
        return self.depth[state][:]

    def get_map(self):
        return self.map

    def all_objects(self):
        if self.using_raw_metadata:
            return [o["objectId"] for o in self.metadata[str(self.state)]["objects"]]
        else:
            return self.metadata.keys()