import os
import pickle
from LMG import LMG
from dataset.locomotion_dataset import LOCODataset
from scripts.args import parse_transfer_loco_opt
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

    source_dataset = LOCODataset(
        data_path=opt.data_path,
        backup_path=opt.processed_data_dir,
        train=False,
        force_reload=opt.force_reload,
        data_sub=opt.source,
    )

    output_scaler = source_dataset.output_scaler
    input_scaler = source_dataset.input_scaler
    data_pipe = source_dataset.data_pipe
    
    for motion_ in source_dataset.motion_list:
        all_motion.append(motion_)
    
    for ctrl_ in source_dataset.ctrl_list:
        all_cond.append(ctrl_)
        all_filenames.append(opt.source)

    print(len(all_motion))
    print(len(all_cond))
    print(len(all_filenames))
    
    source_model = LMG(opt.checkpoint_source)
    source_model.eval()
    target_model = LMG(opt.checkpoint_target)
    target_model.eval()

    render_out = os.path.join(opt.render_dir, "0_transfer_"+opt.source+"_to_"+opt.target)

    all_score = 0
    print("Transfering dances")
    for i in range(0, len(all_cond), 10):
        # source to latent
        source = all_motion[i].unsqueeze(0).to(source_model.accelerator.device)

        data_tuple = source, all_cond[i].unsqueeze(0), all_filenames[i]
        source_model.render_sample_source(
            data_tuple, f"{i}or", render_out, output_scaler, input_scaler, data_pipe
        )

        cond =all_cond[i].unsqueeze(0).to(source_model.accelerator.device)
        latent = source_model.diffusion.reverse_ddim_sample(source, cond)

        # latent to target
        data_tuple = latent, all_cond[i].unsqueeze(0), all_filenames[i]
        target = target_model.render_sample(
            data_tuple, f"{i}tr", render_out, output_scaler, input_scaler, data_pipe
        )

        # target back latent
        target = target.to(target_model.accelerator.device)
        latent_r = target_model.diffusion.reverse_ddim_sample(target, cond)

        # latent to source_back
        data_tuple = latent_r, all_cond[i].unsqueeze(0), all_filenames[i]
        source_re = source_model.render_sample(
            data_tuple, f"{i}cy", render_out, output_scaler, input_scaler, data_pipe
        )
        source = source.detach().cpu()
        score = cal_cyc(source, source_re)


        print(all_filenames[i])
        print(score)
        all_score += score
        
        if i > 100:
            avg_score = all_score / 12 / 150
            print(avg_score)
            return
    print("Done")
    torch.cuda.empty_cache()


if __name__ =="__main__":
    # seed_everything(134)

    opt = parse_transfer_loco_opt()
    transfer(opt)