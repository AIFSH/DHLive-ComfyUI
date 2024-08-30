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

import torch
from PIL import Image
from torchvision.transforms.functional import normalize
from basicsr.utils import imwrite, img2tensor, tensor2img
from basicsr.utils.download_util import load_file_from_url 
from basicsr.utils.registry import ARCH_REGISTRY

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
                            "RestoreFormer++","PGTFormer","KEEP"],),
                "padding":("INT",{
                    "default": 512,
                }),
                "if_RIFE":("BOOLEAN",{
                    "default":True
                }),
            }
        }
    
    RETURN_TYPES = ("VIDEO",)
    #RETURN_NAMES = ("image_output_name",)

    FUNCTION = "generate"

    #OUTPUT_NODE = False

    CATEGORY = "AIFSH_DHLive"

    def generate(self,audio,video,if_data_preparation,upscale,padding,if_RIFE):
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
        if if_RIFE:
            import math
            def get_64x_num(num):
                return math.ceil(num / 64) * 64
            height,width  = (1024,get_64x_num(1024*vid_width/vid_height)) if vid_height > vid_width else (get_64x_num(1024*vid_height/vid_width),1024)
        cap_input.release()
        task_id = str(uuid.uuid1())
        output_video_name = os.path.join(ouput_dir,task_id+".mp4")
        task_id_path = os.path.join(ouput_dir,"DHLive","output",task_id)
        os.makedirs(task_id_path, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        save_path = os.path.join(task_id_path,"silence.mp4")
        videoWriter = cv2.VideoWriter(save_path, fourcc, 25, (int(vid_width) * 1, int(vid_height)))
        image_frame_list = []
        if upscale == "PGTFormer":
            import yaml
            with open(os.path.join(checkpoints_dir,"upscale","PGTFormer.yml"), mode='r') as f:
                opt = yaml.load(f, Loader=ordered_yaml()[0])
            ooo = opt['network_g']
            print(ooo)
            # from talkingface.basicsr.archs.pgtformer_arch import PGTFormer
            # upscale_model = PGTFormer(**ooo).cuda()
            upscale_model = ARCH_REGISTRY.get("PGTFormer")(**ooo).cuda()
            state_dict = torch.load(os.path.join(checkpoints_dir,"upscale","PGTFormer.pth"))
            upscale_model.load_state_dict(state_dict=state_dict['params_ema'])
            upscale_model.eval()
            upscale_model.requires_grad_(False)
            # Read the first frame
            frame_buffer = []
            pre_frame = None
            pre_crop_coords = None
            for i , raw_frame in tqdm(enumerate(mouth_frame)):
                frame, face_numpy, crop_coords = renderModel.interface(raw_frame,upscale="None",padding=padding)
                face_frame = cv2.resize(face_numpy, (512, 512), interpolation=cv2.INTER_LINEAR)
                if i == 0:
                    # print(face_frame.shape)
                    frame_buffer.append(face_frame)
                    frame_buffer.append(face_frame)
                else:
                    frame_buffer.append(face_frame)
                    if len(frame_buffer) == 3:
                        processed_frame = apply_net_to_frames(frame_buffer, upscale_model)
                        # processed_face_numpy = cv2.resize(processed_frame, (256,256), interpolation=cv2.INTER_LINEAR)
                        x_min, y_min, x_max, y_max = pre_crop_coords

                        img_face = cv2.resize(processed_frame, (x_max - x_min, y_max - y_min),interpolation=cv2.INTER_LINEAR)
                        
                        pre_frame[y_min:y_max, x_min:x_max] = img_face
                        videoWriter.write(pre_frame)
                        if if_RIFE:
                            pre_frame =  cv2.resize(pre_frame, (width, height),interpolation=cv2.INTER_LINEAR)
                            image_frame_list.append(Image.fromarray(cv2.cvtColor(pre_frame,cv2.COLOR_BGR2RGB)))
                        # Remove the first frame from the buffer and continue
                        frame_buffer.pop(0)
                pre_crop_coords = crop_coords
                pre_frame = frame
            
            # Process the last two frames (when padding a frame is needed)
            if len(frame_buffer) == 2:
                frame_buffer.append(frame_buffer[-1])  # Pad the last frame
                processed_frame = apply_net_to_frames(frame_buffer, upscale_model)
                # processed_face_numpy = cv2.resize(processed_frame, (256,256), interpolation=cv2.INTER_LINEAR)
                x_min, y_min, x_max, y_max = pre_crop_coords
                img_face = cv2.resize(processed_frame, (x_max - x_min, y_max - y_min),interpolation=cv2.INTER_LINEAR)
                pre_frame[y_min:y_max, x_min:x_max] = img_face
                videoWriter.write(pre_frame)
                if if_RIFE:
                    pre_frame =  cv2.resize(pre_frame, (width, height),interpolation=cv2.INTER_LINEAR)
                    image_frame_list.append(Image.fromarray(cv2.cvtColor(pre_frame,cv2.COLOR_BGR2RGB)))

        else:
            cropped_faces = []
            crop_coords = []
            frames_list = []
            for frame in tqdm(mouth_frame):
                if upscale == "KEEP":
                    frame,face_img,coords = renderModel.interface(frame,upscale="None",padding=padding)
                    face_img = cv2.resize(face_img, (512, 512), interpolation=cv2.INTER_LINEAR)
                    cropped_face_t = img2tensor(
                        face_img / 255., bgr2rgb=True, float32=True)
                    normalize(cropped_face_t, (0.5, 0.5, 0.5),
                                (0.5, 0.5, 0.5), inplace=True)
                    cropped_faces.append(cropped_face_t)
                    crop_coords.append(coords)
                    frames_list.append(frame)
                else:
                    frame,face_img,coords = renderModel.interface(frame,upscale=upscale,padding=padding)
                    videoWriter.write(frame)
                if if_RIFE:
                    frame =  cv2.resize(frame, (width, height),interpolation=cv2.INTER_LINEAR)
                    image_frame_list.append(Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)))
            if upscale == "KEEP":
                # ------------------ set up restorer -------------------
                net = ARCH_REGISTRY.get('KEEP')(img_size=512, emb_dim=256, dim_embd=512,
                                n_head=8, n_layers=9, codebook_size=1024, connect_list=['16', '32', '64'],
                                flow_type='gmflow', flownet_path=None,
                                kalman_attn_head_dim=48, num_uncertainty_layers=3, cross_fuse_list=['16', '32'],
                                cross_fuse_nhead=4, cross_fuse_dim=256).to("cuda")

                # ckpt_path = 'weights/KEEP/KEEP-a1a14d46.pth'
                # checkpoint = torch.load(ckpt_path)['params_ema']

                ckpt_path = load_file_from_url(
                    url='https://github.com/jnjaby/KEEP/releases/download/v0.1.0/KEEP-a1a14d46.pth',
                    model_dir=os.path.join(checkpoints_dir,"upscale"), progress=True, file_name=None)
                checkpoint = torch.load(ckpt_path)
                net.load_state_dict(checkpoint)
                net.eval()

                cropped_faces = torch.stack(
                    cropped_faces, dim=0).unsqueeze(0).to("cuda")
                print(cropped_faces.shape)
                with torch.no_grad():
                    video_length = cropped_faces.shape[1]
                    output = []
                    for start_idx in range(0, video_length, 20):
                        end_idx = min(start_idx + 20, video_length)
                        if end_idx - start_idx == 1:
                            output.append(net(
                                cropped_faces[:, [start_idx, start_idx], ...], need_upscale=False)[:, 0:1, ...])
                        else:
                            output.append(net(
                                cropped_faces[:, start_idx:end_idx, ...], need_upscale=False))
                    output = torch.cat(output, dim=1).squeeze(0)
                    assert output.shape[0] == video_length, "Differer number of frames"

                    restored_faces = [tensor2img(
                        x, rgb2bgr=True, min_max=(-1, 1)) for x in output]
                    del output
                    torch.cuda.empty_cache()
                for frame,face,coords in tqdm(zip(frames_list,restored_faces,crop_coords)):

                    x_min, y_min, x_max, y_max = coords
                    img_face = cv2.resize(face, (x_max - x_min, y_max - y_min),interpolation=cv2.INTER_LINEAR)
                    frame[y_min:y_max, x_min:x_max] = img_face
                    videoWriter.write(frame)

        videoWriter.release()
        if if_RIFE:
            import imageio
            from .rife import IFNet,RIFESmoother
            model = IFNet().eval()
            state_dict = torch.load(os.path.join(checkpoints_dir,"upscale","flownet.pkl"),map_location="cpu")
            model.load_state_dict(model.state_dict_converter().from_civitai(state_dict))
            model.to(torch.float32).to("cuda")
            smoother = RIFESmoother.load_model(model)
            out_video = smoother(image_frame_list)
            print("use RIFE flow")
            def save_video(frames, save_path, fps=25, quality=9):
                writer = imageio.get_writer(save_path, fps=fps, quality=quality)
                for frame in tqdm(frames, desc="Saving video"):
                    frame = np.array(frame)
                    writer.append_data(frame)
                writer.close()
            save_video(out_video,save_path)
            
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