from my_diffusers.schedulers import DDPMScheduler, DDIMScheduler, DPMSolverMultistepScheduler, EulerDiscreteScheduler, UniPCMultistepScheduler

class_dict = {"DDPM": DDPMScheduler,
              "DDIM": DDIMScheduler,
              "DPM": DPMSolverMultistepScheduler,
              "Euler": EulerDiscreteScheduler,
              "UniPC": UniPCMultistepScheduler}


def get_diffusion_scheduler(config, name="DDPM"):
    assert name in ("DDPM", "DDIM")
    adaptive_betas = config.get("adaptive_betas", False)
    if not adaptive_betas or config.get("dyn_scheduler", False):
        noise_scheduler = class_dict[name].from_pretrained(config.pretrained_model_name_or_path,
                                                           rescale_betas_zero_snr=config.rescale_betas_zero_snr,
                                                           subfolder="scheduler",
                                                           local_files_only=True,
                                                           beta_schedule=config.get("beta_schedule", "scaled_linear"),
                                                           snr_rescale=config.get("snr_rescale", 1.0),
                                                           dyn_scheduler=config.get("dyn_scheduler", False))
    else:
        assert len(config.snr_rescale) == config.nframe // 2  # max conditions == nframe/2
        noise_scheduler = dict()
        for f in range(config.nframe // 2, config.nframe):
            noise_scheduler[f] = class_dict[name].from_pretrained(config.pretrained_model_name_or_path,
                                                                  rescale_betas_zero_snr=config.rescale_betas_zero_snr,
                                                                  subfolder="scheduler",
                                                                  local_files_only=True,
                                                                  beta_schedule="snr_rescale",
                                                                  snr_rescale=config.snr_rescale[f - (config.nframe // 2)],
                                                                  dyn_scheduler=config.get("dyn_scheduler", False))

    return noise_scheduler
