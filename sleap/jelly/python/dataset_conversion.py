import sleap
import numpy as np
import os
from copy import copy
from typing import Sequence

def copy_suggestions_func(src: sleap.Labels, dst: sleap.Labels) -> None:
    if not src.suggestions:
        return
    new_suggestions = []
    for suggestion in src.suggestions:
        new_suggestion = sleap.gui.suggestions.SuggestionFrame(video=dst.video, frame_idx=suggestion.frame_idx, group=suggestion.group)
        new_suggestions.append(new_suggestion)
    dst.suggestions = new_suggestions

def single2multi(single_animal_dataset: sleap.Labels, 
                 assign_track: bool=True, 
                 copy_suggestions: bool=True, 
                 exclude_missing_frames: bool=False,
                 save_path: str=None) -> sleap.Labels:
    
    tb_cnt = len(single_animal_dataset.skeletons[0].nodes) - 1 # 1 mouth; the rest are tentacles
    print(f'Tentacle count: {tb_cnt}')
    new_skeleton = sleap.Skeleton(name=f'TB')
    new_skeleton.add_node(f'tb')
    
    multi_animal_dataset = copy(single_animal_dataset)
    multi_animal_dataset.skeletons = [new_skeleton]
    
    if assign_track:
        all_tracks = [sleap.instance.Track(name=f'track_{i}', spawned_on=0) for i in range(tb_cnt)] # assume all tb are present in frame 0
        multi_animal_dataset.tracks = all_tracks
    
    new_labeled_frames = []
    mouth_locations = []
    for lf in single_animal_dataset.labeled_frames:
        # generate new instances 
        new_instances = [None for _ in range(tb_cnt)]
        labeled_inst = None
        for inst in lf.instances: # each frame has at most 2 instances: labeled and predicted
            if isinstance(inst, sleap.Instance) and not isinstance(inst, sleap.PredictedInstance):
                labeled_inst = inst
                break
        if labeled_inst is None or (exclude_missing_frames and len(labeled_inst.nodes) < tb_cnt + 1):
            continue
        for old_node, old_point in zip(labeled_inst.nodes, labeled_inst.points):
            if old_node.name.lower() == 'mouth':
                mouth_locations.append([old_point.x, old_point.y])
                continue
            
            point_dict = {f'tb': sleap.instance.Point(x=old_point.x, y=old_point.y)}
            tb_instance = sleap.Instance(skeleton=new_skeleton, points=point_dict, frame=lf)
            tb_idx = int(old_node.name[2:])
            if assign_track:
                tb_instance.track = all_tracks[tb_idx - 1]
            new_instances[tb_idx - 1] = tb_instance
        
        if None in new_instances:
            # print(f'None indices: {np.where(np.array(new_instances) == None)[0]}')
            new_instances = [inst for inst in new_instances if inst is not None]
            
        new_lf = sleap.LabeledFrame(video=single_animal_dataset.video, frame_idx=lf.frame_idx, instances=new_instances)
        new_labeled_frames.append(new_lf)
    
    multi_animal_dataset.labeled_frames = new_labeled_frames
    
    if copy_suggestions:
        copy_suggestions_func(single_animal_dataset, multi_animal_dataset)
    
    if save_path is not None:
        parent_dir = os.path.dirname(save_path)
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)
        multi_animal_dataset.save(save_path)
    
    mouth_locations = np.array(mouth_locations)
    print(f'Average mouth location: {np.mean(mouth_locations, axis=0)}')
    
    return multi_animal_dataset

def multi2single(multi_animal_dataset: sleap.Labels, 
                 mouth_location: Sequence[float]=(86.19, 79.87),
                 copy_suggestions: bool=True, 
                 exclude_missing_frames: bool=True,
                 save_path: str=None) -> sleap.Labels:
    
    assert multi_animal_dataset.tracks, 'Tracks are not assigned'
    single_animal_dataset = copy(multi_animal_dataset)
    tb_cnt = len(multi_animal_dataset.tracks)
    print(f'Tentacle count: {tb_cnt}')
    
    new_skeleton = sleap.Skeleton(name=f'jellyfish')
    new_skeleton.add_node(f'Mouth')
    for i in range(tb_cnt):
        new_skeleton.add_node(f'TB{i+1}')
        new_skeleton.add_edge(f'Mouth', f'TB{i+1}')
    single_animal_dataset.skeletons = [new_skeleton]
    
    new_labeled_frames = []
    for lf in multi_animal_dataset.labeled_frames:
        # generate new instances 
        instances = [inst for inst in lf.instances if not isinstance(inst, sleap.PredictedInstance) and isinstance(inst, sleap.Instance)]
        if exclude_missing_frames and len(instances) < tb_cnt:
            continue
        point_dict = {f'Mouth': sleap.instance.Point(x=mouth_location[0], y=mouth_location[1])}
        curr_tb_cnt = 0
        
        for inst in instances:
            old_point = inst.points[0]
            tb_idx = int(inst.track.name[6:]) + 1
            point_dict[f'TB{tb_idx}'] = sleap.instance.Point(x=old_point.x, y=old_point.y)
            curr_tb_cnt += 1
            
        jellyfish_inst = sleap.Instance(skeleton=new_skeleton, points=point_dict, frame=lf)
            
        if curr_tb_cnt != tb_cnt:
            print(f'TB count mismatch at frame {lf.frame_idx}: {curr_tb_cnt} != {tb_cnt}')
            
        new_lf = sleap.LabeledFrame(video=single_animal_dataset.video, frame_idx=lf.frame_idx, instances=[jellyfish_inst])
        new_labeled_frames.append(new_lf)
    
    single_animal_dataset.labeled_frames = new_labeled_frames
    single_animal_dataset.tracks = []
    
    if copy_suggestions:
        copy_suggestions_func(multi_animal_dataset, single_animal_dataset)
    
    if save_path is not None:
        parent_dir = os.path.dirname(save_path)
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)
        single_animal_dataset.save(save_path)
    
    return single_animal_dataset
        
    