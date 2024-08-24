import os
import torch
import numpy as np
import folder_paths
now_dir = os.path.dirname(os.path.abspath(__file__))
input_dir = folder_paths.get_input_directory()
output_dir = folder_paths.get_output_directory()
from moviepy.audio.AudioClip import AudioArrayClip
from moviepy.editor import VideoFileClip,CompositeAudioClip,afx,AudioFileClip

class LoadVideo:
    @classmethod
    def INPUT_TYPES(s):
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f)) and f.split('.')[-1] in ["mp4", "webm","mkv","avi"]]
        return {"required":{
            "video":(files,),
        }}
    
    CATEGORY = "AIFSH_DHLive"
    DESCRIPTION = "hello world!"

    RETURN_TYPES = ("VIDEO","AUDIO",)

    OUTPUT_NODE = False

    FUNCTION = "load_video"

    def load_video(self, video):
        video_path = os.path.join(input_dir,video)
        video_clip = VideoFileClip(video_path)
        print(video_clip.audio.fps)
        print(video_clip.audio.duration)
        audio_np = []
        for chunk in video_clip.audio.iter_chunks(chunk_duration=video_clip.audio.duration):
            audio_np.append(chunk)
        audio_np = np.concatenate(audio_np,0)
        audio_np = audio_np.transpose(1,0)
        audio = {
            "waveform": torch.from_numpy(audio_np).unsqueeze(0),
            "sample_rate": video_clip.audio.fps
        }
        
        return (video_path,audio,)

class CombineVideo:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":{
            "video":("VIDEO",),
            "audio": ("AUDIO",),
            "volumn":("FLOAT",{
                "default": 1.0
            })
        }}
    
    CATEGORY = "AIFSH_DHLive"
    DESCRIPTION = "hello world!"

    RETURN_TYPES = ("VIDEO",)

    OUTPUT_NODE = False

    FUNCTION = "combine_video"

    def combine_video(self,video,audio,volumn):
        video_clip = VideoFileClip(video)
        video_clip.audio
        waveform = audio['waveform'].numpy()[0]
        fps = audio['sample_rate']
        waveform = waveform.transpose(1,0)
        bgm_clip = afx.audio_loop(AudioArrayClip(waveform,fps),duration=video_clip.duration)
        audio_clip = CompositeAudioClip([video_clip.audio, bgm_clip.volumex(volumn)])
        video_clip = video_clip.set_audio(audio_clip)
        video_path = os.path.join(output_dir,"combine_"+ os.path.basename(video))
        video_clip.write_videofile(video_path)
        return (video_path,)


class PreViewVideo:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":{
            "video":("VIDEO",),
        }}
    
    CATEGORY = "AIFSH_DHLive"
    DESCRIPTION = "hello world!"

    RETURN_TYPES = ()

    OUTPUT_NODE = True

    FUNCTION = "load_video"

    def load_video(self, video):
        video_name = os.path.basename(video)
        video_path_name = os.path.basename(os.path.dirname(video))
        return {"ui":{"video":[video_name,video_path_name]}}
