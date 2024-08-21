Run command:
- For DMC: python train.py task=quadruped_walk expl_cfg=icm expl_cfg.use_my_ratio=true
- For PixMC: python train.py task=FrankaCabinetSparsePixels expl_cfg=icm expl_cfg.use_my_ratio=true
- For MiniGrid: python src/train.py --int_rew_source=RND --model_features_dim=64 --model_latents_dim=64 --game_name=RedBlueDoors-8x8 --use_my_ratio=True

Among them, 'use_my_ratio' means using the AIRES method. If you encounter any problems during operation, please feel free to contact me.
