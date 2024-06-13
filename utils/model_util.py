from diffusion import gaussian_diffusion as gd
from diffusion.respace import SpacedDiffusion, space_timesteps

from utils.fixseed import fixseed

def create_diffusion():
    diffusion = create_gaussian_diffusion(get_default_diffusion())
    return diffusion


def get_default_diffusion():
    args = {
        "noise_schedule": "cosine",
        "sigma_small": True,
    }
    return args


def get_model_args():
    return {
        "arch": "trans_enc",
        "batch_size": 64,
        "cond_mask_prob": 0.1,
        "cuda": True,
        "data_dir": "",
        "dataset": "humanml",
        "device": 0,
        "diffusion_steps": 1000,
        "emb_trans_dec": False,
        "eval_batch_size": 32,
        "eval_during_training": False,
        "eval_num_samples": 1000,
        "eval_rep_times": 3,
        "eval_split": "test",
        "lambda_fc": 0.0,
        "lambda_rcxyz": 0.0,
        "lambda_vel": 0.0,
        "lambda_cat": 0.05,
        "latent_dim": 512,
        "layers": 8,
        "log_interval": 1000,
        "lr": 0.0001,
        "lr_anneal_steps": 0,
        "noise_schedule": "cosine",
        "num_frames": 60,
        "num_steps": 600000,
        "overwrite": False,
        "resume_checkpoint": "",
        "save_dir": "save/my_humanml_trans_enc_512",
        "save_interval": 50000,
        "seed": 10,
        "sigma_small": True,
        "train_platform_type": "NoPlatform",
        "unconstrained": False,
        "weight_decay": 0.0
    }

def create_gaussian_diffusion(args):
    # default params
    predict_xstart = True  # we always predict x_start (a.k.a. x0), that's our deal!
    steps = 1000
    scale_beta = 1.  # no scaling
    timestep_respacing = ''  # can be used for ddim sampling, we don't use it.
    learn_sigma = False
    rescale_timesteps = False

    betas = gd.get_named_beta_schedule(args['noise_schedule'], steps)
    loss_type = gd.LossType.MSE

    if not timestep_respacing:
        timestep_respacing = [steps]

    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not args['sigma_small']
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
    )

def get_training_platform():
    args = {
        'seed': 10,
        'train_platform_type': "NoPlatform",
        'save_dir': "debug/"
    }

    fixseed(args['seed'])
    train_platform_type = eval(args['train_platform_type'])
    train_platform = train_platform_type(args['save_dir'])
