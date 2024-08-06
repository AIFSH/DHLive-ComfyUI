import os,sys
import uuid
from .data_preparation import CirculateVideo,work_dir,now_dir,ouput_dir
sys.path.append(now_dir)

import cv2
import shutil
import torchaudio
from tqdm import tqdm
from PIL import Image
import numpy as np
from talkingface.render_model import RenderModel
from talkingface.audio_model import AudioModel

checkpoints_dir = os.path.join(now_dir,"checkpoint")

class StaticVideo:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "image":("IMAGE",),
                "length":("INT",{
                    "default":3
                })
            }
        }
    
    RETURN_TYPES = ("VIDEO",)
    #RETURN_NAMES = ("image_output_name",)

    FUNCTION = "generate"

    #OUTPUT_NODE = False

    CATEGORY = "AIFSH_DHLive"

    def comfyimage2Image(self,comfyimage):
        comfyimage = comfyimage.numpy()[0] * 255
        image_np = comfyimage.astype(np.uint8)
        image = Image.fromarray(image_np)
        return image

    def generate(self,image,length):
        image = self.comfyimage2Image(image)
        image_path = os.path.join(work_dir,"source.jpg")
        image.save(image_path)
        video_path = os.path.join(work_dir,"video.mp4")
        cmd = f"ffmpeg -r 25 -f image2 -loop 1 -i {image_path} -vcodec libx264 -pix_fmt yuv420p -r 25 -t {length} -y {video_path}"
        os.system(cmd)
        return (video_path,)

prompt_sr = 16000
class DHLiveNode:

    def __init__(self) -> None:
        self.audioModel = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "audio":("AUDIO",),
                "video":("VIDEO",),
                "if_data_preparation":("BOOLEAN",{
                    "default":True
                })
            }
        }
    
    RETURN_TYPES = ("VIDEO",)
    #RETURN_NAMES = ("image_output_name",)

    FUNCTION = "generate"

    #OUTPUT_NODE = False

    CATEGORY = "AIFSH_DHLive"

    def generate(self,audio,video,if_data_preparation):
        ## 1. data preparation
        video_data_path = os.path.join(work_dir,"video_data")
        if if_data_preparation:
            os.makedirs(video_data_path,exist_ok=True)
            video_out_path = os.path.join(video_data_path,"circle.mp4")
            CirculateVideo(video_in_path=video,video_out_path=video_out_path)

        ## 
        if self.audioModel is None:
            self.audioModel = AudioModel()
            self.audioModel.loadModel(os.path.join(checkpoints_dir,"audio.pkl"))

            self.renderModel = RenderModel()
            self.renderModel.loadModel(os.path.join(checkpoints_dir,"render.pth"))
        else:
            self.audioModel.reset()
        pkl_path = "{}/keypoint_rotate.pkl".format(video_data_path)
        video_path = "{}/circle.mp4".format(video_data_path)
        self.renderModel.reset_charactor(video_path, pkl_path)

        waveform = audio['waveform'].squeeze(0)
        source_sr = audio['sample_rate']
        speech = waveform.mean(dim=0,keepdim=True)
        if source_sr != prompt_sr:
            speech = torchaudio.transforms.Resample(orig_freq=source_sr, new_freq=prompt_sr)(speech)
        wavpath = os.path.join(ouput_dir,"tmp.wav")
        torchaudio.save(wavpath,speech,prompt_sr,format="wav")
        mouth_frame = self.audioModel.interface_wav(speech.numpy()[0])

        cap_input = cv2.VideoCapture(video_path)
        vid_width = cap_input.get(cv2.CAP_PROP_FRAME_WIDTH)  # 宽度
        vid_height = cap_input.get(cv2.CAP_PROP_FRAME_HEIGHT)  # 高度
        cap_input.release()
        task_id = str(uuid.uuid1())
        output_video_name = os.path.join(ouput_dir,task_id+".mp4")
        task_id_path = os.path.join(ouput_dir,"DHLive","output",task_id)
        os.makedirs(task_id_path, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        save_path = os.path.join(task_id_path,"silence.mp4")
        videoWriter = cv2.VideoWriter(save_path, fourcc, 25, (int(vid_width) * 1, int(vid_height)))
        for frame in tqdm(mouth_frame):
            frame = self.renderModel.interface(frame)
            # cv2.imshow("s", frame)
            # cv2.waitKey(40)

            videoWriter.write(frame)

        videoWriter.release()
        os.system(
            "ffmpeg -i {} -i {} -c:v libx264 -pix_fmt yuv420p -loglevel quiet {}".format(save_path, wavpath, output_video_name))
        shutil.rmtree(task_id_path)
        return (output_video_name,)

WEB_DIRECTORY = "./web"

from .util_nodes import PreViewVideo, LoadVideo

NODE_CLASS_MAPPINGS = {
    "LoadVideo":LoadVideo,
    "PreViewVideo":PreViewVideo, 
    "DHLiveNode": DHLiveNode,
    "StaticVideo":StaticVideo
}