import os
import subprocess

def reencode_video(input_file: str, output_file: str, ss_str:str=None) -> None:
    """Reencodes video into h.264 coded format using ffmpeg from a subprocess.

    Args:
        input_file: abspath to existing video
        output_file: abspath to to new mp4 video using h.264 codec

    """
    # check input file exists
    assert os.path.isfile(input_file), 'input video does not exist.'
    # check directory for saving outputs exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    # create ffmpeg command
    ffmpeg_cmd = f'ffmpeg -y -i {input_file} -ss {ss_str} -t 60 -c:v libx264 -pix_fmt yuv420p -preset superfast -crf 0 {output_file}'
    # run command
    subprocess.run(ffmpeg_cmd, shell=True)

if __name__ == "__main__":
    # for i in [2, 3]:
    #     INPUT_FILE = f"/home/mingxiao/Desktop/jellyfish/video/full_video_{i}.avi"
    #     OUTPUT_FILE = f"/home/mingxiao/Desktop/jellyfish/video/sleap_full_video_{i}_higher_res.mp4"
    ss_strs = ['00:00:19', '00:01:19', '00:02:19', '00:03:19', '00:04:19']
    for i in range(len(ss_strs)):
        INPUT_FILE = '/home/mingxiao/Desktop/jellyfish/video/video_1_clips/c1_high_res_10min_track_reencoded.mp4'
        OUTPUT_FILE = f'/home/mingxiao/Desktop/jellyfish/video/video_1_clips/c1_high_res_5min_track_reencoded_{i}.mp4'
        reencode_video(INPUT_FILE, OUTPUT_FILE, ss_str=ss_strs[i])