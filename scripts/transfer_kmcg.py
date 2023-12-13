import os
import pickle
from EDGE import EDGE
from dataset.dance_dataset import AISTPPDataset
from scripts.args import parse_transfer_opt
import numpy as np
import torch
import random

def seed_everything(seed=134):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def cal_cyc(source, source_bar):
    
    distance = np.linalg.norm(source - source_bar, axis=-1).mean()
    # length = 
    return distance

def transfer(opt):
    all_motion = []
    all_cond = []
    all_filenames = []

    source_dataset = AISTPPDataset(
        data_path=opt.data_path,
        backup_path=opt.processed_data_dir,
        train=True,
        force_reload=opt.force_reload,
        data_sub=opt.source,
    )
    for pose_ in source_dataset.data["pose"]:
        all_motion.append(pose_)
    
    for filename_ in source_dataset.data["filenames"]:
        cond = torch.from_numpy(np.load(filename_))
        all_cond.append(cond)

    for wavs_ in source_dataset.data["wavs"]:
        all_filenames.append(wavs_)

    print(len(all_motion))
    print(len(all_cond))
    print(len(all_filenames))
    
    source_model = EDGE(opt.feature_type, opt.checkpoint_source)
    source_model.eval()

    target_model = EDGE(opt.feature_type, opt.checkpoint_target)
    target_model.eval()

    fk_out = os.path.join(opt.render_dir, "5_transfer_"+opt.source+"_to_"+opt.target)
    render_out = os.path.join(opt.render_dir, "5_transfer_"+opt.source+"_to_"+opt.target)
    all_score = 0
    print("Transfering dances")
    for i in range(0, len(all_cond), 10):
        # source to latent
        source = all_motion[i].unsqueeze(0).to(source_model.accelerator.device)
        key_indices = get_keyframe_index(source, 5)
        # TODO: save source render
        data_tuple = source, all_cond[i].unsqueeze(0), [all_filenames[i]]
        source_model.render_sample_source(
            data_tuple, f"{i}", render_out, render_count=-1, fk_out=fk_out, render=True
        )

        cond = all_cond[i].unsqueeze(0).to(source_model.accelerator.device)
        latent = source_model.diffusion.reverse_ddim_sample(source, cond)

        # latent to target
        data_tuple = latent, all_cond[i].unsqueeze(0), [all_filenames[i]]
        constraint = {
            'value': source,
            'mask': torch.zeros_like(source)  
        }
        constraint['mask'][:, key_indices, :] = 1. 

        target = target_model.render_sample_inpaint(
            data_tuple, f"{i}", render_out, render_count=-1, fk_out=fk_out, render=True,
            mode = 'inpaint', constraint=constraint
        )

        # target back latent
        target = target.to(target_model.accelerator.device)
        latent_r = target_model.diffusion.reverse_ddim_sample(target, cond)

        # latent to source_back
        key_indices = get_keyframe_index(target, 5)
        constraint = {
            'value': target,
            'mask': torch.zeros_like(target)  
        }
        constraint['mask'][:, key_indices, :] = 1.   

        data_tuple = latent_r, all_cond[i].unsqueeze(0), [all_filenames[i]]
        source_re = source_model.render_sample_inpaint(
            data_tuple, f"{i}r", render_out, render_count=-1, fk_out=fk_out, render=True,
            mode = 'inpaint', constraint=constraint
        )
        source = source.detach().cpu()
        score = cal_cyc(source, source_re)


        print(all_filenames[i])
        print(score)
        all_score += score
        
        if i > 100:
            avg_score = all_score / 12.0 /150.0
            print(avg_score)
            
            from filelock import FileLock

            lock = FileLock("my_lock.lock")

            with lock:
                with open('cn.txt', 'a') as f:  
                    f.write(str(avg_score) + '\n')
            return

        
    print("Done")
    torch.cuda.empty_cache()


if __name__ =="__main__":
    # seed_everything(134)

    opt = parse_transfer_opt()
    transfer(opt)