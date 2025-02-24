import sleap
import numpy as np

def construct_model(raw_model: sleap.Labels, tb_cnt: int=17) -> sleap.Labels:
    skeleton = sleap.Skeleton(name=f'TB')
    skeleton.add_node(f'tb')
    raw_model.skeletons = [skeleton]
    