import argparse

def parse_train_stable_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default="runs/train", help="project/name")
    parser.add_argument("--exp_name", default="exp", help="save to project/name")
    parser.add_argument("--data_path", type=str, default="data/", help="raw data path")
    parser.add_argument("--data_sub", type=str, default="all", help="sub set of the dataset")
    parser.add_argument(
        "--processed_data_dir",
        type=str,
        default="data/dataset_backups/",
        help="Dataset backup path",
    )
    parser.add_argument(
        "--render_dir", type=str, default="renders/", help="Sample render path"
    )

    parser.add_argument("--feature_type", type=str, default="jukebox")
    parser.add_argument(
        "--wandb_pj_name", type=str, default="ddst", help="project name"
    )
    parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    parser.add_argument("--eval_batch_size", type=int, default=2, help="eval batch size")
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument(
        "--force_reload", action="store_true", help="force reloads the datasets"
    )
    parser.add_argument(
        "--no_cache", action="store_true", help="don't reuse / cache loaded dataset"
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=100,
        help='Log model after every "save_period" epoch',
    )
    parser.add_argument("--ema_interval", type=int, default=1, help="ema every x steps")
    parser.add_argument(
        "--checkpoint", type=str, default="", help="trained checkpoint path (optional)"
    )
    parser.add_argument(
        "--checkpoint_encdec", 
        type=str, 
        default="runs/train/exp_pretrain_test/weights/train-1.pt", 
        help="trained checkpoint path encdec"
    )
    parser.add_argument(
        "--conditional", action="store_true", help="pretrain conditional"
    )
    parser.add_argument(
        "--finetune", action="store_true", help="finetune"
    )
    parser.add_argument("--nframe_latent", type=int, default=10, help="nf latent")
    parser.add_argument("--latent_dim", type=int, default=256, help="latent size")
    opt = parser.parse_args()
    return opt

def parse_train_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default="runs/train", help="project/name")
    parser.add_argument("--exp_name", default="exp", help="save to project/name")
    parser.add_argument("--data_path", type=str, default="data/", help="raw data path")
    parser.add_argument("--data_sub", type=str, default="all", help="sub set of the dataset")
    parser.add_argument(
        "--processed_data_dir",
        type=str,
        default="data/dataset_backups/",
        help="Dataset backup path",
    )
    parser.add_argument(
        "--render_dir", type=str, default="renders/", help="Sample render path"
    )

    parser.add_argument("--feature_type", type=str, default="jukebox")
    parser.add_argument(
        "--wandb_pj_name", type=str, default="ddst", help="project name"
    )
    parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    
    parser.add_argument("--eval_batch_size", type=int, default=2, help="eval batch size")
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument(
        "--force_reload", action="store_true", help="force reloads the datasets"
    )
    parser.add_argument(
        "--no_cache", action="store_true", help="don't reuse / cache loaded dataset"
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=100,
        help='Log model after every "save_period" epoch',
    )
    parser.add_argument("--ema_interval", type=int, default=1, help="ema every x steps")
    parser.add_argument(
        "--checkpoint", type=str, default="", help="trained checkpoint path (optional)"
    )
    opt = parser.parse_args()
    return opt


def parse_train_lmd_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default="runs/train", help="project/name")
    parser.add_argument("--exp_name", default="exp", help="save to project/name")
    parser.add_argument("--data_path", type=str, default="data/justLMD/", help="raw data path")
    parser.add_argument("--data_sub", type=str, default="all", help="sub set of the dataset")
    parser.add_argument(
        "--processed_data_dir",
        type=str,
        default="data/dataset_backups/",
        help="Dataset backup path",
    )
    parser.add_argument(
        "--render_dir", type=str, default="renders/", help="Sample render path"
    )

    parser.add_argument("--feature_type", type=str, default="jukebox")
    parser.add_argument(
        "--wandb_pj_name", type=str, default="ddst", help="project name"
    )
    parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    
    parser.add_argument("--eval_batch_size", type=int, default=2, help="eval batch size")
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument(
        "--force_reload", action="store_true", help="force reloads the datasets"
    )
    parser.add_argument(
        "--no_cache", action="store_true", help="don't reuse / cache loaded dataset"
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=100,
        help='Log model after every "save_period" epoch',
    )
    parser.add_argument("--ema_interval", type=int, default=1, help="ema every x steps")
    parser.add_argument(
        "--checkpoint", type=str, default="", help="trained checkpoint path (optional)"
    )
    opt = parser.parse_args()
    return opt


def parse_train_loco_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default="runs/train", help="project/name")
    parser.add_argument("--exp_name", default="exp_loco", help="save to project/name")
    parser.add_argument("--data_path", type=str, default="data/loco_processed_scaled", help="raw data path")
    parser.add_argument("--data_sub", type=str, default="all", help="sub set of the dataset")
    parser.add_argument(
        "--processed_data_dir",
        type=str,
        default="data/dataset_backups/",
        help="Dataset backup path",
    )
    parser.add_argument(
        "--render_dir", type=str, default="renders/", help="Sample render path"
    )
    parser.add_argument("--source", type=str, default="Angry")
    parser.add_argument("--target", type=str, default="Cat")
    parser.add_argument("--feature_type", type=str, default="jukebox")
    parser.add_argument(
        "--wandb_pj_name", type=str, default="ddst", help="project name"
    )
    parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    
    parser.add_argument("--eval_batch_size", type=int, default=2, help="eval batch size")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument(
        "--force_reload", action="store_true", help="force reloads the datasets"
    )
    parser.add_argument(
        "--no_cache", action="store_true", help="don't reuse / cache loaded dataset"
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=1,
        help='Log model after every "save_period" epoch',
    )
    parser.add_argument("--ema_interval", type=int, default=1, help="ema every x steps")
    parser.add_argument(
        "--checkpoint", type=str, default="", help="trained checkpoint path (optional)"
    )
    opt = parser.parse_args()
    return opt



def parse_test_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_type", type=str, default="jukebox")
    parser.add_argument("--out_length", type=float, default=4, help="max. length of output, in seconds")
    parser.add_argument(
        "--processed_data_dir",
        type=str,
        default="data/dataset_backups/",
        help="Dataset backup path",
    )
    parser.add_argument("--data_sub", type=str, default="all", help="sub set of the dataset")
    parser.add_argument(
        "--render_dir", type=str, default="renders/", help="Sample render path"
    )
    parser.add_argument(
        "--checkpoint", type=str, default="runs/train/exp_gJB/weights/train-18000.pt", help="checkpoint"
    )
    parser.add_argument(
        "--music_dir",
        type=str,
        default="data/custom_music",
        help="folder containing input music",
    )
    parser.add_argument(
        "--save_motions", action="store_true", help="Saves the motions for evaluation"
    )
    parser.add_argument(
        "--motion_save_dir",
        type=str,
        default="eval/motions",
        help="Where to save the motions",
    )
    parser.add_argument(
        "--cache_features",
        action="store_true",
        help="Save the jukebox features for later reuse",
    )
    parser.add_argument(
        "--no_render",
        action="store_true",
        help="Don't render the video",
    )
    parser.add_argument(
        "--use_cached_features",
        action="store_true",
        help="Use precomputed features instead of music folder",
    )
    parser.add_argument(
        "--feature_cache_dir",
        type=str,
        default="data/custom/",
        help="Where to save/load the features",
    )
    opt = parser.parse_args()
    return opt

def parse_transfer_loco_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_type", type=str, default="jukebox")
    parser.add_argument("--out_length", type=float, default=4, help="max. length of output, in seconds")
    parser.add_argument("--source", type=str, default="Drunk")
    parser.add_argument("--target", type=str, default="Robot")
    parser.add_argument("--data_path", type=str, default="data/loco_processed_scaled", help="raw data path")
    parser.add_argument(
        "--processed_data_dir",
        type=str,
        default="data/dataset_backups/",
        help="Dataset backup path",
    )
    parser.add_argument(
        "--checkpoint_source", type=str, default="runs/train/exp_Drunk2/weights/train-6000.pt", help="checkpoint source"
    )
    parser.add_argument(
        "--checkpoint_target", type=str, default="runs/train/exp_Robot2/weights/train-3000.pt", help="checkpoint target"
    )
    parser.add_argument(
        "--force_reload", action="store_true", help="force reloads the datasets"
    )
    parser.add_argument(
        "--render_dir", type=str, default="renders/", help="Sample render path"
    )
    opt = parser.parse_args()
    return opt

def parse_transfer_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_type", type=str, default="jukebox")
    parser.add_argument("--out_length", type=float, default=4, help="max. length of output, in seconds")
    parser.add_argument("--source", type=str, default="gLO")
    parser.add_argument("--target", type=str, default="gKR")
    parser.add_argument("--data_path", type=str, default="data/", help="raw data path")
    parser.add_argument(
        "--processed_data_dir",
        type=str,
        default="data/dataset_backups/",
        help="Dataset backup path",
    )
    parser.add_argument(
        "--checkpoint_source", type=str, default="runs/train/exp_gLO/weights/train-18000.pt", help="checkpoint source"
    )
    parser.add_argument(
        "--checkpoint_target", type=str, default="runs/train/exp_gKR/weights/train-18000.pt", help="checkpoint target"
    )
    parser.add_argument(
        "--force_reload", action="store_true", help="force reloads the datasets"
    )
    parser.add_argument(
        "--render_dir", type=str, default="renders/", help="Sample render path"
    )
    opt = parser.parse_args()
    return opt

def parse_transfer_stable_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_type", type=str, default="jukebox")
    parser.add_argument("--out_length", type=float, default=4, help="max. length of output, in seconds")
    parser.add_argument("--source", type=str, default="gLO")
    parser.add_argument("--target", type=str, default="gKR")
    parser.add_argument("--data_path", type=str, default="data/", help="raw data path")
    parser.add_argument(
        "--processed_data_dir",
        type=str,
        default="data/dataset_backups/",
        help="Dataset backup path",
    )
    parser.add_argument(
        "--checkpoint_encdec", type=str, default="runs/train/exp_gLO/weights/train-18000.pt", help="checkpoint source"
    )
    parser.add_argument(
        "--checkpoint_source", type=str, default="runs/train/exp_gLO/weights/train-18000.pt", help="checkpoint source"
    )
    parser.add_argument(
        "--checkpoint_target", type=str, default="runs/train/exp_gKR/weights/train-18000.pt", help="checkpoint target"
    )
    parser.add_argument(
        "--force_reload", action="store_true", help="force reloads the datasets"
    )
    parser.add_argument(
        "--render_dir", type=str, default="renders/", help="Sample render path"
    )
    parser.add_argument(
        "--conditional", action="store_true", help="pretrain conditional"
    )
    parser.add_argument(
        "--finetune", action="store_true", help="finetune"
    )
    parser.add_argument("--nframe_latent", type=int, default=150, help="nf latent")
    parser.add_argument("--latent_dim", type=int, default=128, help="latent size")

    opt = parser.parse_args()
    return opt


def parse_pretrain_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default="runs/train", help="project/name")
    parser.add_argument("--exp_name", default="exp_pretrain", help="save to project/name")
    parser.add_argument("--data_path", type=str, default="data/", help="raw data path")
    parser.add_argument("--data_sub", type=str, default="all", help="sub set of the dataset")
    parser.add_argument(
        "--processed_data_dir",
        type=str,
        default="data/dataset_backups/",
        help="Dataset backup path",
    )
    parser.add_argument(
        "--render_dir", type=str, default="renders/", help="Sample render path"
    )

    parser.add_argument("--feature_type", type=str, default="jukebox")
    parser.add_argument(
        "--wandb_pj_name", type=str, default="ddst", help="project name"
    )
    parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    parser.add_argument("--nframe_latent", type=int, default=10, help="nf latent")
    parser.add_argument("--latent_dim", type=int, default=256, help="latent size")
    parser.add_argument("--eval_batch_size", type=int, default=2, help="eval batch size")
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument(
        "--force_reload", action="store_true", help="force reloads the datasets"
    )
    parser.add_argument(
        "--no_cache", action="store_true", help="don't reuse / cache loaded dataset"
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=100,
        help='Log model after every "save_period" epoch',
    )
    parser.add_argument(
        "--checkpoint", type=str, default="", help="trained checkpoint path (optional)"
    )
    parser.add_argument(
        "--conditional", action="store_true", help="pretrain conditional"
    )
    opt = parser.parse_args()
    return opt