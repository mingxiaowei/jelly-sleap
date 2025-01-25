import sleap
import numpy as np
import matplotlib.pyplot as plt

def get_point_location(labels_path, frame_index):
    labels = sleap.load_file(labels_path)
    frame1 = labels.video.get_frame(frame_index)
    
    # Version 2: Hover to get coordinates
    def onmove(event):
        if event.inaxes == ax:
            ax.format_coord = lambda x, y: f'x={int(x)}, y={int(y)}'
    
    # Display the image
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(frame1[:,:,0], cmap='gray')
    cid = fig.canvas.mpl_connect('motion_notify_event', onmove)
    plt.show()

if __name__ == "__main__":
    labels_path = '/home/mingxiao/Desktop/jellyfish/video/video_1_clips/predictions/predictions/c3_predictions.slp.250120_125305.predictions.slp'
    frame_index = 1
    get_point_location(labels_path, frame_index)