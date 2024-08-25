import os
import torch
import numpy as np
import folder_paths
now_dir = os.path.dirname(os.path.abspath(__file__))
input_dir = folder_paths.get_input_directory()
output_dir = folder_paths.get_output_directory()
import torchaudio
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
        tmp_file = os.path.join(output_dir,"tmp.wav")
        video_clip.audio.write_audiofile(tmp_file)
        '''
        audio_np = []
        for chunk in video_clip.audio.iter_chunks(chunk_duration=video_clip.audio.duration):
            audio_np.append(chunk)
        audio_np = np.concatenate(audio_np,0)
        audio_np = audio_np.transpose(1,0)
        '''
        waveform, sample_rate = torchaudio.load(tmp_file)
        audio = {
            "waveform": waveform.unsqueeze(0),#torch.from_numpy(audio_np).unsqueeze(0),
            "sample_rate": sample_rate
        }
        
        return (video_path,audio,)

class CombineVideo:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "video":("VIDEO",),
                "keep_org_audio":("BOOLEAN",{
                    "default": True,
                }),
                
            },
            "optional":{
                "audio": ("AUDIO",),
                "bgm_audio": ("AUDIO",),
                "bgm_volumn":("FLOAT",{
                    "default": 1.0
                })
            }
        }
    
    CATEGORY = "AIFSH_DHLive"
    DESCRIPTION = "hello world!"

    RETURN_TYPES = ("VIDEO",)

    OUTPUT_NODE = False

    FUNCTION = "combine_video"

    def combine_video(self,video,keep_org_audio,audio=None,bgm_audio=None,bgm_volumn=1.0):
        video_clip = VideoFileClip(video)
        
        audio_clip_list = []
        if not keep_org_audio:
            video_clip = video_clip.without_audio()
        else:
            audio_clip_list.append(video_clip.audio)

        if audio:
            waveform = audio['waveform'].numpy()[0]
            fps = audio['sample_rate']
            waveform = waveform.transpose(1,0)
            audio_clip_list.append(afx.audio_loop(AudioArrayClip(waveform,fps),duration=video_clip.duration))
        if bgm_audio:
            waveform = bgm_audio['waveform'].numpy()[0]
            fps = bgm_audio['sample_rate']
            waveform = waveform.transpose(1,0)
            bgm_audio_clip = afx.audio_loop(AudioArrayClip(waveform,fps),duration=video_clip.duration)
            audio_clip_list.append(bgm_audio_clip.volumex(bgm_volumn))
        
        audio_clip = CompositeAudioClip(audio_clip_list)
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
