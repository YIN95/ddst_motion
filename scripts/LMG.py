import multiprocessing
import os
import pickle
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
import wandb
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.state import AcceleratorState
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.locomotion_dataset import LOCODataset
from dataset.preprocess import increment_path
from model.adan import Adan
from model.diffusion_loco import GaussianDiffusion
from model.model import DanceDecoder
from scripts.vis import SMPLSkeleton


def wrap(x):
    return {f"module.{key}": value for key, value in x.items()}


def maybe_wrap(x, num):
    return x if num == 1 else wrap(x)


class LMG:
    def __init__(
        self,
        checkpoint_path="",
        normalizer=None,
        EMA=True,
        learning_rate=4e-4,
        weight_decay=0.02,
    ):
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
        state = AcceleratorState()
        num_processes = state.num_processes

        self.repr_dim = repr_dim = 72

        feature_dim = 3

        horizon_seconds = 5
        FPS = 30
        self.horizon = horizon = horizon_seconds * FPS

        self.accelerator.wait_for_everyone()

        checkpoint = None
        if checkpoint_path != "":
            checkpoint = torch.load(
                checkpoint_path, map_location=self.accelerator.device
            )
            # self.normalizer = checkpoint["normalizer"]

        model = DanceDecoder(
            nfeats=repr_dim,
            seq_len=horizon,
            latent_dim=512,
            ff_size=1024,
            num_layers=8,
            num_heads=8,
            dropout=0.1,
            cond_feature_dim=feature_dim,
            activation=F.gelu,
        )
        smpl = SMPLSkeleton(self.accelerator.device)
        diffusion = GaussianDiffusion(
            model,
            horizon,
            repr_dim,
            smpl,
            schedule="cosine",
            n_timestep=1000,
            predict_epsilon=False,
            loss_type="l2",
            use_p2=False,
            cond_drop_prob=0.25,
            guidance_weight=2,
        )

        print(
            "Model has {} parameters".format(sum(y.numel() for y in model.parameters()))
        )

        self.model = self.accelerator.prepare(model)
        self.diffusion = diffusion.to(self.accelerator.device)
        optim = Adan(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.optim = self.accelerator.prepare(optim)

        if checkpoint_path != "":
            # self.model.load_state_dict(
            #     maybe_wrap(
            #         checkpoint["ema_state_dict" if EMA else "model_state_dict"],
            #         num_processes,
            #     )
            # )
            self.model.load_state_dict(
                maybe_wrap(
                    checkpoint["dk"],
                    num_processes,
                )
            )

        

    def eval(self):
        self.diffusion.eval()

    def train(self):
        self.diffusion.train()

    def prepare(self, objects):
        return self.accelerator.prepare(*objects)

    def train_loop(self, opt):
        data_cache_train = f"train_tensor_loco_dataset_{opt.data_sub}"
        data_cache_test = f"test_tensor_loco_dataset_{opt.data_sub}"

        train_tensor_dataset_path = os.path.join(
            opt.processed_data_dir, f"{data_cache_train}.pkl"
        )
        test_tensor_dataset_path = os.path.join(
            opt.processed_data_dir, f"{data_cache_test}.pkl"
        )
        if (
            not opt.force_reload
            and not opt.no_cache
            and os.path.isfile(train_tensor_dataset_path)
            and os.path.isfile(test_tensor_dataset_path)
        ):
            train_dataset = pickle.load(open(train_tensor_dataset_path, "rb"))
            test_dataset = pickle.load(open(test_tensor_dataset_path, "rb"))
        else:
            train_dataset = LOCODataset(
                data_path=opt.data_path,
                backup_path=opt.processed_data_dir,
                train=True,
                force_reload=opt.force_reload,
                data_sub=opt.data_sub,
            )
            test_dataset = LOCODataset(
                data_path=opt.data_path,
                backup_path=opt.processed_data_dir,
                train=False,
                force_reload=opt.force_reload,
                data_sub=opt.data_sub
            )
            # cache the dataset in case
            if self.accelerator.is_main_process:
                pickle.dump(train_dataset, open(train_tensor_dataset_path, "wb"))
                pickle.dump(test_dataset, open(test_tensor_dataset_path, "wb"))
        self.train_dataset = train_dataset
        self.output_scaler = train_dataset.output_scaler
        self.input_scaler = train_dataset.input_scaler
        self.data_pipe = train_dataset.data_pipe
        # data loaders
        # decide number of workers based on cpu count
        num_cpus = multiprocessing.cpu_count()
        train_data_loader = DataLoader(
            train_dataset,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=min(int(num_cpus * 0.75), 32),
            pin_memory=True,
            drop_last=True,
        )
        test_data_loader = DataLoader(
            test_dataset,
            batch_size=opt.eval_batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
            drop_last=True,
        )

        train_data_loader = self.accelerator.prepare(train_data_loader)
        # boot up multi-gpu training. test dataloader is only on main process
        
        if self.accelerator.is_main_process:
            save_dir = str(increment_path(Path(opt.project) / opt.exp_name))
            opt.exp_name = save_dir.split("/")[-1]
            wandb.init(project=opt.wandb_pj_name, name=opt.exp_name, save_code=False, sync_tensorboard=False)
            wandb.config.save = False
            save_dir = Path(save_dir)
            wdir = save_dir / "weights"
            wdir.mkdir(parents=True, exist_ok=True)

        self.accelerator.wait_for_everyone()
        for epoch in range(1, opt.epochs + 1):
            # if epoch == 10:
            #     for name, param in self.model.named_parameters():
            #         print(f'Parameter: {name}')
            #         print(param)
            #         print()
            load_loop = (
                partial(tqdm, position=1, desc=f"Epoch-{epoch}")
                if self.accelerator.is_main_process
                else lambda x: x
            )
            avg_loss = 0
            avg_vloss = 0
            avg_fkloss = 0
            avg_footloss = 0
            # train
            self.train()
            for step, (x, cond, label, style) in enumerate(
                load_loop(train_data_loader)
            ):
                total_loss, (loss, v_loss, fk_loss, foot_loss) = self.diffusion(
                    x, cond, t_override=None
                )
                self.optim.zero_grad()
                self.accelerator.backward(total_loss)

                self.optim.step()

                # ema update and train loss update only on main
                if self.accelerator.is_main_process:
                    avg_loss += loss.detach().cpu().numpy()
                    avg_vloss += v_loss.detach().cpu().numpy()
                    avg_fkloss += fk_loss.detach().cpu().numpy()
                    avg_footloss += foot_loss.detach().cpu().numpy()
                    if step % opt.ema_interval == 0:
                        self.diffusion.ema.update_model_average(
                            self.diffusion.master_model, self.diffusion.model
                        )
            # Save model
            if (epoch % opt.save_interval) == 0:
                # everyone waits here for the val loop to finish ( don't start next train epoch early)
                self.accelerator.wait_for_everyone()
                # save only if on main thread
                if self.accelerator.is_main_process:
                    self.eval()
                    # log
                    avg_loss /= len(train_data_loader)
                    avg_vloss /= len(train_data_loader)
                    avg_fkloss /= len(train_data_loader)
                    avg_footloss /= len(train_data_loader)
                    log_dict = {
                        "Train Loss": avg_loss,
                        "V Loss": avg_vloss,
                        "FK Loss": avg_fkloss,
                        "Foot Loss": avg_footloss,
                    }
                    wandb.log(log_dict)
                    ckpt = {
                        "ema_state_dict": self.diffusion.master_model.state_dict(),
                        "model_state_dict": self.accelerator.unwrap_model(
                            self.model
                        ).state_dict(),
                        "optimizer_state_dict": self.optim.state_dict(),
                        "dk": self.diffusion.model.state_dict(),
                    }
                    torch.save(ckpt, os.path.join(wdir, f"train-{epoch}.pt"))
                    # generate a sample
                    render_count = opt.eval_batch_size
                    shape = (render_count, self.horizon, self.repr_dim)
                    print("Generating Sample")
                    # draw a music from the test dataset
                    (x, cond, label, style) = next(iter(test_data_loader))
                    cond = cond.to(self.accelerator.device)
                    self.diffusion.render_sample(
                        shape,
                        cond[:render_count],
                        self.output_scaler,
                        self.input_scaler, 
                        self.data_pipe, 
                        os.path.join(opt.render_dir, "train_" + opt.exp_name),
                        name=f'{opt.data_sub}_e{epoch}',
                    )
                    print(f"[MODEL SAVED at Epoch {epoch}]")
        # if self.accelerator.is_main_process:
        #     wandb.run.finish()

    def render_sample(
        self, data_tuple, label, render_dir, output_scaler, input_scaler, data_pipe
    ):
        latent, cond, fname = data_tuple
        assert len(cond.shape) == 3
     
        shape = (1, self.horizon, self.repr_dim)
        cond = cond.to(self.accelerator.device)
        sr = self.diffusion.render_sample(
            shape,
            cond,
            output_scaler,
            input_scaler, 
            data_pipe, 
            render_dir,
            name=fname+label,
            # mode="long",
            noise=latent
        )
        return sr

    def render_sample_source(
        self, data_tuple, label, render_dir, output_scaler, input_scaler, data_pipe
    ):
        sample, cond, fname = data_tuple
        assert len(cond.shape) == 3
        
        cond = cond.to(self.accelerator.device)
        self.diffusion.render_sample_source(
            sample,
            cond,
            output_scaler,
            input_scaler,
            data_pipe,
            render_out=render_dir,
            name=fname+label,
            mode="long",
        )