import os.path
import torch
import os,sys
import numpy as np
import time
from talkingface.run_utils import smooth_array, video_pts_process
from talkingface.run_utils import mouth_replace, prepare_video_data
from talkingface.utils import generate_face_mask, INDEX_LIPS_OUTER
from talkingface.data.few_shot_dataset import select_ref_index,get_ref_images_fromVideo,generate_input, generate_input_pixels
device = "cuda" if torch.cuda.is_available() else "cpu"
import pickle
import cv2

face_mask = generate_face_mask()
now_dir = os.path.dirname(os.path.abspath(__file__))
ckpt_dir = os.path.join(os.path.dirname(now_dir),"checkpoint","upscale")
sys.path.append(now_dir)

class RenderModel:
    def __init__(self):
        self.__net = None

        self.__pts_driven = None
        self.__mat_list = None
        self.__pts_normalized_list = None
        self.__face_mask_pts = None
        self.__ref_img = None
        self.__cap_input = None
        self.frame_index = 0
        self.__mouth_coords_array = None
        self.model_name = None

    def loadModel(self, ckpt_path):
        from talkingface.models.DINet import DINet_five_Ref as DINet
        n_ref = 5
        source_channel = 6
        ref_channel = n_ref * 6
        self.__net = DINet(source_channel, ref_channel).cuda()
        checkpoint = torch.load(ckpt_path)
        self.__net.load_state_dict(checkpoint)
        self.__net.eval()

    def reset_charactor(self, video_path, Path_pkl, ref_img_index_list = None):
        if self.__cap_input is not None:
            self.__cap_input.release()

        self.__pts_driven, self.__mat_list,self.__pts_normalized_list, self.__face_mask_pts, self.__ref_img, self.__cap_input = \
            prepare_video_data(video_path, Path_pkl, ref_img_index_list)

        ref_tensor = torch.from_numpy(self.__ref_img / 255.).float().permute(2, 0, 1).unsqueeze(0).cuda()
        self.__net.ref_input(ref_tensor)

        x_min, x_max = np.min(self.__pts_normalized_list[:, INDEX_LIPS_OUTER, 0]), np.max(self.__pts_normalized_list[:, INDEX_LIPS_OUTER, 0])
        y_min, y_max = np.min(self.__pts_normalized_list[:, INDEX_LIPS_OUTER, 1]), np.max(self.__pts_normalized_list[:, INDEX_LIPS_OUTER, 1])
        z_min, z_max = np.min(self.__pts_normalized_list[:, INDEX_LIPS_OUTER, 2]), np.max(self.__pts_normalized_list[:, INDEX_LIPS_OUTER, 2])

        x_mid,y_mid,z_mid = (x_min + x_max)/2, (y_min + y_max)/2, (z_min + z_max)/2
        x_len, y_len, z_len = (x_max - x_min)/2, (y_max - y_min)/2, (z_max - z_min)/2
        x_min, x_max = x_mid - x_len*0.9, x_mid + x_len*0.9
        y_min, y_max = y_mid - y_len*0.9, y_mid + y_len*0.9
        z_min, z_max = z_mid - z_len*0.9, z_mid + z_len*0.9

        # print(face_personal.shape, x_min, x_max, y_min, y_max, z_min, z_max)
        coords_array = np.zeros([100, 150, 4])
        for i in range(100):
            for j in range(150):
                coords_array[i, j, 0] = j/149
                coords_array[i, j, 1] = i/100
                # coords_array[i, j, 2] = int((-75 + abs(j - 75))*(2./3))
                coords_array[i, j, 2] = ((j - 75)/ 75) ** 2
                coords_array[i, j, 3] = 1

        coords_array = coords_array*np.array([x_max - x_min, y_max - y_min, z_max - z_min, 1]) + np.array([x_min, y_min, z_min, 0])
        self.__mouth_coords_array = coords_array.reshape(-1, 4).transpose(1, 0)



    def interface(self, mouth_frame,upscale):
        vid_frame_count = self.__cap_input.get(cv2.CAP_PROP_FRAME_COUNT)
        if self.frame_index % vid_frame_count == 0:
            self.__cap_input.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 设置要获取的帧号
        ret, frame = self.__cap_input.read()  # 按帧读取视频

        epoch = self.frame_index // len(self.__mat_list)
        if epoch % 2 == 0:
            new_index = self.frame_index % len(self.__mat_list)
        else:
            new_index = -1 - self.frame_index % len(self.__mat_list)

        # print(self.__face_mask_pts.shape, "ssssssss")
        source_img, target_img, crop_coords = generate_input_pixels(frame, self.__pts_driven[new_index], self.__mat_list[new_index],
                                                                    mouth_frame, self.__face_mask_pts[new_index],
                                                                    self.__mouth_coords_array)

        # tensor
        source_tensor = torch.from_numpy(source_img / 255.).float().permute(2, 0, 1).unsqueeze(0).cuda()
        target_tensor = torch.from_numpy(target_img / 255.).float().permute(2, 0, 1).unsqueeze(0).cuda()

        source_tensor, source_prompt_tensor = source_tensor[:, :3], source_tensor[:, 3:]
        fake_out = self.__net.interface(source_tensor, source_prompt_tensor)

        image_numpy = fake_out.detach().squeeze(0).cpu().float().numpy()
        image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0
        image_numpy = image_numpy.clip(0, 255)
        image_numpy = image_numpy.astype(np.uint8)

        image_numpy = target_img * face_mask + image_numpy * (1 - face_mask)
        
        if upscale != "None":
            upscale_image_numpy = self.face_upscale(upscale,image_numpy)
        else:
            upscale_image_numpy = image_numpy

        img_bg = frame
        x_min, y_min, x_max, y_max = crop_coords

        img_face = cv2.resize(upscale_image_numpy, (x_max - x_min, y_max - y_min))
        
        img_bg[y_min:y_max, x_min:x_max] = img_face
        self.frame_index += 1
        return img_bg

    def save(self, path):
        torch.save(self.__net.state_dict(), path)
    

    def face_upscale(self,model_name,img_face):
        import torch
        from torchvision.transforms.functional import normalize
        from basicsr.utils import imwrite, img2tensor, tensor2img
        from basicsr.utils.download_util import load_file_from_url
        from basicsr.utils.registry import ARCH_REGISTRY
        if model_name=="GFPGANv1.4":
            pretrain_model_url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth' 
            if self.model_name != model_name:
                self.upscale_net=ARCH_REGISTRY.get('GFPGANv1Clean')(out_size=512,
                num_style_feat=512,
                channel_multiplier=2,
                decoder_load_path=None,
                fix_decoder=False,
                num_mlp=8,
                input_is_latent=True,
                different_w=True,
                narrow=1,
                sft_half=True).to(device)
                ckpt_path = load_file_from_url(url=pretrain_model_url, 
                                            model_dir=ckpt_dir, progress=True, file_name=None)
                loadnet = torch.load(ckpt_path)
                if 'params_ema' in loadnet:
                    keyname = 'params_ema'
                else:
                    keyname = 'params'
                checkpoint = loadnet[keyname]
                self.upscale_net.load_state_dict(checkpoint)
                self.upscale_net.eval()
        elif model_name == "CodeFormer":
            pretrain_model_url = 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer_inpainting.pth' 
            if self.model_name != model_name:
                self.upscale_net = ARCH_REGISTRY.get('CodeFormer')(dim_embd=512, codebook_size=512, n_head=8, n_layers=9, 
                                                    connect_list=['32', '64', '128']).to(device)
                ckpt_path = load_file_from_url(url=pretrain_model_url, 
                                            model_dir=ckpt_dir, progress=True, file_name=None)
                checkpoint = torch.load(ckpt_path)['params_ema']
                self.upscale_net.load_state_dict(checkpoint)
                self.upscale_net.eval()
        elif model_name == "RestoreFormer++":
            pretrain_model_url = "https://github.com/wzhouxiff/RestoreFormerPlusPlus/releases/download/v1.0.0/RestoreFormer++.ckpt"
            if self.model_name != model_name:
                self.upscale_net = ARCH_REGISTRY.get("VQVAEGANMultiHeadTransformer")(head_size = 4, ex_multi_scale_num = 1).to(device)

                ckpt_path = load_file_from_url(url=pretrain_model_url, 
                                            model_dir=ckpt_dir, progress=True, file_name=None)
                loadnet = torch.load(ckpt_path)
                weights = loadnet['state_dict']
                new_weights = {}
                for k, v in weights.items():
                    if k.startswith('vqvae.'):
                        k = k.replace('vqvae.', '')
                    new_weights[k] = v
                self.upscale_net.load_state_dict(new_weights, strict=False)
                self.upscale_net.eval()
        
        self.model_name = model_name

        org_w,org_h = img_face.shape[:2]
        img_face = cv2.resize(img_face, (512, 512), interpolation=cv2.INTER_LINEAR)
        input_face = img2tensor(img_face / 255., bgr2rgb=True, float32=True)
        normalize(input_face, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
        input_face = input_face.unsqueeze(0).to(device)

        try:
            with torch.no_grad():
                if self.model_name == "CodeFormer":
                    mask = torch.zeros(512, 512)
                    m_ind = torch.sum(input_face[0], dim=0)
                    mask[m_ind==3] = 1.0
                    mask = mask.view(1, 1, 512, 512).to(device)
                    # w is fixed to 1, adain=False for inpaintin
                    output_face = self.upscale_net(input_face, w=1, adain=False)[0]
                    output_face = (1-mask)*input_face + mask*output_face
                elif self.model_name == "GFPGANv1.4":
                    output_face = self.upscale_net(input_face, return_rgb=False, weight=0.5)[0]
                    output_face = output_face.squeeze(0)
                elif self.model_name == "RestoreFormer++":
                    output_face = self.upscale_net(input_face)[0]
                    output_face = output_face.squeeze(0)
                save_face = tensor2img(output_face, rgb2bgr=True, min_max=(-1, 1))
            del output_face
            torch.cuda.empty_cache()
        except Exception as error:
            print(f'\tFailed inference for {self.model_name}: {error}')
            save_face = tensor2img(input_face, rgb2bgr=True, min_max=(-1, 1))
        
        save_face = save_face.astype('uint8')
        save_face = cv2.resize(save_face,(org_w,org_h),interpolation=cv2.INTER_LINEAR)
        return save_face

