import os,sys
import uuid
from .data_preparation import CirculateVideo,work_dir,now_dir,ouput_dir
sys.path.append(now_dir)
import basicsr
sys.modules["basicsr"] = basicsr
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

import torch
def rgbnp2tensor(rgbnplist):
    rgbnps = np.array(rgbnplist).copy()
    lqinput = np.array(np.array(rgbnps) / 255.0, np.float32)  # [t,h,w,c]
    lqinput = torch.from_numpy(lqinput).permute(0, 3, 1, 2).cuda()
    return lqinput

def apply_net_to_frames(frames, model, w=1.0):
    lqinput = rgbnp2tensor(frames)
    with torch.no_grad():
        restored_faces = model(lqinput, w=w)[0][1]
    restored_faces = torch.clamp(restored_faces, 0, 1)
    restored_face = restored_faces.detach().cpu().permute(1, 2, 0).numpy() * 255
    npface = np.array(restored_face, np.uint8)
    return npface

import yaml
from collections import OrderedDict

def ordered_yaml():
    """Support OrderedDict for yaml.

    Returns:
        yaml Loader and Dumper.
    """
    try:
        from yaml import CDumper as Dumper
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Dumper, Loader

    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper

prompt_sr = 16000
class DHLiveNode:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "audio":("AUDIO",),
                "video":("VIDEO",),
                "if_data_preparation":("BOOLEAN",{
                    "default":True
                }),
                "upscale":(["None","GFPGANv1.4","CodeFormer",
                            "RestoreFormer++","PGTFormer"],)
            }
        }
    
    RETURN_TYPES = ("VIDEO",)
    #RETURN_NAMES = ("image_output_name",)

    FUNCTION = "generate"

    #OUTPUT_NODE = False

    CATEGORY = "AIFSH_DHLive"

    def generate(self,audio,video,if_data_preparation,upscale):
        ## 1. data preparation
        video_data_path = os.path.join(work_dir,"video_data")
        if if_data_preparation:
            shutil.rmtree(video_data_path,ignore_errors=True)
            os.makedirs(video_data_path,exist_ok=True)
            video_out_path = os.path.join(video_data_path,"circle.mp4")
            CirculateVideo(video_in_path=video,video_out_path=video_out_path)
        
        audioModel = AudioModel()
        audioModel.loadModel(os.path.join(checkpoints_dir,"audio.pkl"))

        renderModel = RenderModel()
        renderModel.loadModel(os.path.join(checkpoints_dir,"render.pth"))

        pkl_path = "{}/keypoint_rotate.pkl".format(video_data_path)
        video_path = "{}/circle.mp4".format(video_data_path)
        renderModel.reset_charactor(video_path, pkl_path)

        waveform = audio['waveform'].squeeze(0)
        source_sr = audio['sample_rate']
        speech = waveform.mean(dim=0,keepdim=True)
        if source_sr != prompt_sr:
            speech = torchaudio.transforms.Resample(orig_freq=source_sr, new_freq=prompt_sr)(speech)
        wavpath = os.path.join(ouput_dir,"tmp.wav")
        torchaudio.save(wavpath,speech,prompt_sr,format="wav")
        mouth_frame = audioModel.interface_wav(speech.numpy()[0])

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
        if upscale == "PGTFormer":
            import yaml
            import torch
            with open(os.path.join(checkpoints_dir,"upscale","PGTFormer.yml"), mode='r') as f:
                opt = yaml.load(f, Loader=ordered_yaml()[0])
            ooo = opt['network_g']
            print(ooo)
            # from talkingface.basicsr.archs.pgtformer_arch import PGTFormer
            
            from basicsr.utils.registry import ARCH_REGISTRY
            # upscale_model = PGTFormer(**ooo).cuda()
            upscale_model = ARCH_REGISTRY.get("PGTFormer")(**ooo).cuda()
            state_dict = torch.load(os.path.join(checkpoints_dir,"upscale","PGTFormer.pth"))
            upscale_model.load_state_dict(state_dict=state_dict['params_ema'])
            upscale_model.eval()
            upscale_model.requires_grad_(False)
            # Read the first frame
            frame_buffer = []
            pre_crop_coords = None
            for i , raw_frame in tqdm(enumerate(mouth_frame)):
                frame, face_numpy, crop_coords = renderModel.interface(raw_frame,upscale="None")
                face_frame = cv2.resize(face_numpy, (512, 512), interpolation=cv2.INTER_LINEAR)
                if i == 0:
                    # print(face_frame.shape)
                    frame_buffer.append(face_frame)
                    frame_buffer.append(face_frame)
                else:
                    frame_buffer.append(face_frame)
                    if len(frame_buffer) == 3:
                        processed_frame = apply_net_to_frames(frame_buffer, upscale_model)
                        processed_face_numpy = cv2.resize(processed_frame, (256,256), interpolation=cv2.INTER_LINEAR)
                        x_min, y_min, x_max, y_max = pre_crop_coords

                        img_face = cv2.resize(processed_face_numpy, (x_max - x_min, y_max - y_min))
                        
                        frame[y_min:y_max, x_min:x_max] = img_face
                        videoWriter.write(frame)
                        # Remove the first frame from the buffer and continue
                        frame_buffer.pop(0)
                pre_crop_coords = crop_coords
            
            # Process the last two frames (when padding a frame is needed)
            if len(frame_buffer) == 2:
                frame_buffer.append(frame_buffer[-1])  # Pad the last frame
                processed_frame = apply_net_to_frames(frame_buffer, upscale_model)
                processed_face_numpy = cv2.resize(processed_frame, (256,256), interpolation=cv2.INTER_LINEAR)
                x_min, y_min, x_max, y_max = pre_crop_coords
                img_face = cv2.resize(processed_face_numpy, (x_max - x_min, y_max - y_min))
                frame[y_min:y_max, x_min:x_max] = img_face
                videoWriter.write(frame)
        else:
            for frame in tqdm(mouth_frame):
                frame,_,_ = renderModel.interface(frame,upscale=upscale)
                videoWriter.write(frame)

        videoWriter.release()
        os.system(
            "ffmpeg -i {} -i {} -c:v libx264 -pix_fmt yuv420p -loglevel quiet {}".format(save_path, wavpath, output_video_name))
        shutil.rmtree(task_id_path)
        return (output_video_name,)

WEB_DIRECTORY = "./web"

from .util_nodes import PreViewVideo, LoadVideo,CombineVideo

NODE_CLASS_MAPPINGS = {
    "CombineVideo":CombineVideo,
    "PreViewVideo":PreViewVideo, 
    "DHLiveNode": DHLiveNode,
    "StaticVideo":StaticVideo,
    "DHLIVELoadVideo":LoadVideo,
}