import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tqdm import tqdm
from src.superglue.matching import Matching
from src.superglue.futils import make_matching_plot_fast
from src.georef_helper import resize_to_max_dim, frame2tensor, rotate_nocrop, gray2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = {
    'superpoint': {
        'nms_radius': 4,
        'keypoint_threshold': 0.005,
        'max_keypoints': -1
    },
    'superglue': {
        'weights': 'outdoor',
        'sinkhorn_iterations': 20,
        'match_threshold': 0.2,
    }
}

matching = Matching(config).eval().to(device)
keys = ['keypoints', 'scores', 'descriptors']

class Homography():
    def __init__(self, img_source, img_target):
        """
        Defines a homography between two outdoor images with potential
        temporal differences, for example, lighting and seasonal.
        
        Parameters:
            img_source (ndarray): Gray or RGB image.
            img_target (ndarray): Gray or RGB image.
            match_output (str): File path to output match plot. None if no output.
        """
        TAR_SM_DIM = 480
        SRC_SM_DIM = TAR_SM_DIM
        
        img_source_sm_gr = gray2(resize_to_max_dim(img_source, SRC_SM_DIM))
        img_target_sm_gr = gray2(resize_to_max_dim(img_target, TAR_SM_DIM))
        
        best_transforms = {'ang': 0, 'src': SRC_SM_DIM,
                           'tar': TAR_SM_DIM, 'met': 0}
        
        ########################################################
        ##### ANGLE SEARCH #####################################
        ########################################################
        
        frame_tensor = frame2tensor(img_source_sm_gr, device)

        for ang in tqdm(range(0, 360, 60)):
            img_target_sm_gr_rot, rot_M = rotate_nocrop(img_target_sm_gr, ang)
            image_tensor = frame2tensor(img_target_sm_gr_rot, device)
            
            m_kwargs, conf_valid = self.match(frame_tensor, image_tensor)
            mkpts0 = m_kwargs['mkpts0']
            
            if np.sum(conf_valid)/(1+len(mkpts0))**0.5 > best_transforms['met']:
                best_transforms['ang'] = ang
                best_transforms['met'] = np.sum(conf_valid)/(1+len(mkpts0))**0.5
        
        print("Rotation search complete.", best_transforms)
        
        ########################################################
        ##### SCALE SEARCH #####################################
        ########################################################
        
        for s_dim in tqdm((320, 480, 640, 800, 960)):
            img_source_sm_gr = gray2(resize_to_max_dim(img_source, s_dim))
            frame_tensor = frame2tensor(img_source_sm_gr, device)            
            
            for t_dim in (480, 640, 800, 960, 1280):
                img_target_sm_gr = gray2(resize_to_max_dim(img_target, t_dim))
                img_target_sm_gr_rot, rot_M = rotate_nocrop(img_target_sm_gr, ang)
                image_tensor = frame2tensor(img_target_sm_gr_rot, device)
                
                m_kwargs, conf_valid = self.match(frame_tensor, image_tensor)
                mkpts0 = m_kwargs['mkpts0']
                
                if np.sum(conf_valid)/(1+len(mkpts0))**0.5 > best_transforms['met']:
                    best_transforms['src'] = s_dim
                    best_transforms['tar'] = t_dim
                    best_transforms['met'] = np.sum(conf_valid)/(1+len(mkpts0))**0.5
                    
        print("Scale search complete.", best_transforms)
                
        ########################################################        
        ##### USING BEST ROTATION AND SCALE ####################
        ########################################################
        
        img_source_lg, ss = resize_to_max_dim(img_source, best_transforms['src'], return_scale=True)
        img_target_lg, st = resize_to_max_dim(img_target, best_transforms['tar'], return_scale=True)
        img_source_lg_gr = gray2(img_source_lg)
        img_target_lg_gr = gray2(img_target_lg)
        img_target_lg_gr_rot, rot_M = rotate_nocrop(img_target_lg_gr, best_transforms['ang'])

        frame_tensor = frame2tensor(img_source_lg_gr, device)
        image_tensor = frame2tensor(img_target_lg_gr_rot, device)
        kpt_kwargs, conf_valid = self.match(frame_tensor, image_tensor)
        color = cm.jet(conf_valid)        
        
        self.out = make_matching_plot_fast(
            img_source_lg_gr, img_target_lg_gr_rot, kpt_kwargs['kpts0'],
            kpt_kwargs['kpts1'],kpt_kwargs['mkpts0'],kpt_kwargs['mkpts1'],
            color, text=[], show_keypoints=True)
        
        ########################################################        
        ##### HOMOGRAPHY CALCULATION ###########################
        ########################################################

        if len(mkpts0) < 10: # in the future, make this a manual thing? click and drag?
            raise Exception
        hom_M, _ = cv2.findHomography(kpt_kwargs['mkpts0'], kpt_kwargs['mkpts1'], cv2.RANSAC, 5.0)
        h,w,*_ = img_target_lg_gr_rot.shape
        img_source_warp = cv2.warpPerspective(img_source_lg, hom_M, (w, h))
        h,w,*_ = img_target_lg_gr.shape
        img_source_warp = cv2.warpAffine(img_source_warp, rot_M, (w,h), flags=cv2.WARP_INVERSE_MAP)
        mask = (img_source_warp > 0)
        
        self.overlay = img_target_lg.copy()
        self.overlay[mask] = img_source_warp[mask]
        self.best_transforms = best_transforms
        rot_M = np.vstack((rot_M, (0,0,1)))
        self.HOM_M = np.diag([1/st, 1/st, 1]) @ np.linalg.inv(rot_M) @ hom_M @ np.diag([ss,ss,1])
        
    @staticmethod
    def match(tensor_src, tensor_tar):
        last_data = matching.superpoint({'image': tensor_src})
        last_data = {k+'0': last_data[k] for k in keys}
        last_data['image0'] = tensor_src
        
        pred = matching({**last_data, 'image1': tensor_tar})
        kpts0 = last_data['keypoints0'][0].cpu().numpy()
        kpts1 = pred['keypoints1'][0].cpu().numpy()
        matches = pred['matches0'][0].cpu().numpy()
        confidence = pred['matching_scores0'][0].cpu().numpy()

        valid = matches > -1
        mkpts0 = kpts0[valid]
        mkpts1 = kpts1[matches[valid]]
        
        return dict(kpts0=kpts0, kpts1=kpts1,
                    mkpts0=mkpts0, mkpts1=mkpts1), confidence[valid]