import json
import os
import time
import warnings
from collections import deque
from math import gcd
from multiprocessing import Process, Queue

from ai2thor.controller import BFSController


from bfs_controller import ExhaustiveBFSController
# from off_c import ExhaustiveBFSController


def search_and_save(in_queue):
    while not in_queue.empty():
        try:
            scene_name = in_queue.get(timeout=3)
        except:
            return
        c = None
        try:
            out_dir = os.path.join("/media/gregory/data/dump/", scene_name)
            if os.path.exists(out_dir):
                continue
            else:
                os.mkdir(out_dir)

            print('starting:', scene_name)
            c = ExhaustiveBFSController(
                grid_size=0.25,
                fov=63.453048374758716,
                grid_file=os.path.join(out_dir, 'grid.json'),
                graph_file=os.path.join(out_dir, 'graph.json'),
                metadata_file=os.path.join(out_dir, 'metadata.json'),
                images_file=os.path.join(out_dir, 'images.hdf5'),
                depth_file=os.path.join(out_dir, 'depth.hdf5'),
                grid_assumption=False)
            # c.start()
            c.search_all_closed(scene_name)
            c.stop()
        except AssertionError as e:
            print('Error is', e)
            print('Error in scene {}'.format(scene_name))
            if c is not None:
                c.stop()
            continue


def main():

    num_processes = 4
    
    queue = Queue()
    scene_names = []
    
    for i in range(12):
        for j in range(5):
            scene_names.append("FloorPlan_Train{}_{}".format(i+1, j+1))

    # for x in scene_names:
    #     print(x)
    # scene_names.append("FloorPlan_Train1_1")
    # scene_names.append("FloorPlan_Train5_2")
    # scene_names.append("FloorPlan_Train5_3")
    for x in scene_names:
        queue.put(x)

    processes = []
    for _ in range(num_processes):
        p = Process(target=search_and_save, args=(queue,))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

if __name__ == "__main__":
    main()
