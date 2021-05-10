import importlib
from collections import deque
import json
import copy
import time
import random
import os
import platform

# try:
#     from queue import Queue
# except ImportError:
#     from Queue import Queue

from ai2thor.controller import Controller, distance
from base_controller import BaseController
from bfs_controller import ExhaustiveBFSController, ThorAgentState


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
        # metadata_file_name="visible_object_map.json",
        metadata_file_name='metadata.json',
        images_file_name="images.hdf5",
        debug_mode=True,
        actions=["MoveAhead", "RotateLeft", "RotateRight", "LookUp", "LookDown"],
        visualize=True,
        # visualize=False,
        local_executable_path=None
    ):

        super(OfflineControllerWithSmallRotation, self).__init__()
        self.grid_size = grid_size
        self.offline_data_dir = offline_data_dir
        self.grid_file_name = grid_file_name
        self.graph_file_name = graph_file_name
        self.metadata_file_name = metadata_file_name
        self.images_file_name = images_file_name
        self.grid = None
        self.graph = None
        self.metadata = None
        self.images = None
        self.controller = None
        self.using_raw_metadata = True
        self.actions = actions
        # Allowed rotations.
        self.rotations = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
        # Allowed horizons.
        self.horizons = [-30, 0, 30]
        self.debug_mode = debug_mode
        self.fov = fov

        self.local_executable_path = local_executable_path

        self.y = None

        self.last_event = None

        self.controller = ExhaustiveBFSController()
        if self.local_executable_path is not None:
            self.controller.local_executable_path = self.local_executable_path

        self.visualize = visualize

        self.scene_name = None
        self.state = None
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
        print("executing action: {}".format(action))

        next_state = self.controller.get_next_state(self.state, action, True)
        print("next state: {}".format(next_state))

        if self.visualize and next_state is not None:
            viz_event = self.controller.step(
                dict(
                    action="Teleport", 
                    x=next_state.x, 
                    y=next_state.y, 
                    z=next_state.z, 
                    rotation=next_state.rotation, 
                    horizon=next_state.horizon
                )
            )
            # viz_event = self.controller.step(
            #     dict(action="Rotate", rotation=next_state.rotation)
            # )
            # viz_event = self.controller.step(
            #     dict(action="Look", horizon=next_state.horizon)
            # )
            viz_next_state = self.controller.get_state_from_event(viz_event)
            if (
                round(viz_next_state.horizon) not in self.horizons
                or round(viz_next_state.rotation) not in self.rotations
            ):
                # return back to original state.
                self.controller.teleport_to_state(self.state)

        if next_state is not None:
            next_state_key = str(next_state)
            neighbors = list(self.graph.neighbors(str(self.state)))

            if next_state_key in neighbors:
                self.state = self.get_state_from_str(
                    *[float(x) for x in next_state_key.split("|")]
                )
                self.last_action_success = True
                event = self._successful_event()
                if self.debug_mode and self.visualize:
                    if self.controller.get_state_from_event(
                        viz_event
                    ) != self.controller.get_state_from_event(event):
                        print(action)
                        print(str(self.controller.get_state_from_event(viz_event)))
                        print(str(self.controller.get_state_from_event(event)))

                    assert self.controller.get_state_from_event(
                        viz_event
                    ) == self.controller.get_state_from_event(event)
                    assert viz_event.metadata["lastActionSuccess"]

                    # Uncomment if you want to view the frames side by side to
                    # ensure that they are duplicated.
                    from matplotlib import pyplot as plt
                    fig = plt.figure()
                    fig.add_subplot(2,1,1)
                    plt.imshow(self.get_image())
                    fig.add_subplot(2,1,2)
                    plt.imshow(viz_event.frame)
                    plt.show()

                self.last_event = event
                return event

        self.last_action_success = False
        self.last_event.metadata["lastActionSuccess"] = False
        return self.last_event

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

    def all_objects(self):
        if self.using_raw_metadata:
            return [o["objectId"] for o in self.metadata[str(self.state)]["objects"]]
        else:
            return self.metadata.keys()