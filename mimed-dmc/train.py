# -*- coding: utf-8 -*-

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1' # Force MKL to use Intel specific optimizations,
# This is true even on non Intel processors. This can improve performance in certain situations, especially in math intensive applications that use MKL.
os.environ['MUJOCO_GL'] = 'egl' # 'egl' is an interface for OpenGL ES and other graphics APIs that allows rendering without directly relying on the window system

from pathlib import Path

import hydra
import numpy as np
import torch
from dm_env import specs
import sys
import dmc
import utils
from logger import Logger
from replay_buffer import ReplayBufferStorage, make_replay_loader
from video import TrainVideoRecorder, VideoRecorder

import wandb

torch.backends.cudnn.benchmark = True
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
sys.path.append(current_dir)


def make_agent(obs_spec, action_spec, cfg):
    cfg.obs_shape = obs_spec.shape
    cfg.action_shape = action_spec.shape
    return hydra.utils.instantiate(cfg)


class Workspace:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.setup()

        if 'mae' in self.cfg.agent._target_.lower():
            self.cfg.agent.save_dir = f'{self.work_dir}/mae_vis'
            os.makedirs(self.cfg.agent.save_dir, exist_ok=True)

        self.agent = make_agent(self.train_env.observation_spec(),
                                self.train_env.action_spec(),
                                self.cfg.agent)
        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0
        self._best_episode_reward = -np.inf

        if self.cfg.load_ckpt_path != '':
            self.load_ckpt(self.cfg.load_ckpt_path)

    def setup(self):
        # create logger
        self.logger = Logger(self.work_dir, use_tb=self.cfg.use_tb)
        # create envs
        self.train_env = dmc.make(self.cfg.task_name, self.cfg.frame_stack,
                                  self.cfg.action_repeat, self.cfg.seed)
        self.eval_env = dmc.make(self.cfg.task_name, self.cfg.frame_stack,
                                 self.cfg.action_repeat, self.cfg.seed,
                                 pixels_only=False)
        # create replay buffer
        data_specs = (self.train_env.observation_spec(),
                      self.train_env.action_spec(),
                      specs.Array((1,), np.float32, 'reward'),
                      specs.Array((1,), np.float32, 'discount'))

        self.replay_storage = ReplayBufferStorage(data_specs,
                                                  self.work_dir / 'buffer')

        self.replay_loader = make_replay_loader(
            self.work_dir / 'buffer', self.cfg.replay_buffer_size,
            self.cfg.batch_size, self.cfg.replay_buffer_num_workers,
            self.cfg.save_snapshot, self.cfg.nstep, self.cfg.discount,
            return_seq_len=self.cfg.expl_cfg.seq_expl_len,
            return_seq_type=self.cfg.expl_cfg.seq_type,

            return_seq_len_ratio=self.cfg.expl_cfg.expl_seq_len_ratio,
            use_my_ratio=self.cfg.expl_cfg.use_my_ratio
            )
        self._replay_iter = None

        self.video_recorder = VideoRecorder(
            self.work_dir if self.cfg.save_video else None)
        self.train_video_recorder = TrainVideoRecorder(
            self.work_dir if self.cfg.save_train_video else None)

        self.eval_trajs_dir = self.work_dir / 'eval_trajs'
        self.eval_trajs_dir.mkdir(exist_ok=True)

    @property
    def best_episode_reward(self):
        return self._best_episode_reward

    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode

    @property
    def global_frame(self):
        return self.global_step * self.cfg.action_repeat

    @property
    def replay_iter(self):
        if self._replay_iter is None:
            self._replay_iter = iter(self.replay_loader)
        return self._replay_iter

    def eval(self, save_video=True):
        step, episode, total_reward = 0, 0, 0
        eval_until_episode = utils.Until(self.cfg.num_eval_episodes)

        # # log states to plot state occupancy after training
        # states = []

        while eval_until_episode(episode):
            time_step = self.eval_env.reset(return_state=False)
            # states.append(time_step.state)
            self.video_recorder.init(
                self.eval_env, enabled=(save_video and episode == 0))
            while not time_step.last():
                with torch.no_grad(), utils.eval_mode(self.agent):
                    action = self.agent.act(time_step.observation,
                                            self.global_step,
                                            eval_mode=True)
                time_step = self.eval_env.step(action, return_state=False)
                # states.append(time_step.state)
                self.video_recorder.record(self.eval_env)
                total_reward += time_step.reward
                step += 1

            episode += 1
            if save_video:
                self.video_recorder.save(f'{self.global_frame}.mp4')

                if self.video_recorder.enabled:
                    # (T, H, W, C)
                    frames = np.stack(self.video_recorder.frames)
                    frames = frames.transpose((0, 3, 1, 2))
                    # log latest video (T, C, H, W) to wandb
                    wandb.log(
                        {'video': wandb.Video(frames, fps=30, format='mp4')},
                        step=self.global_step)

        # self.logger.save_list_of_dict(
        #     states, self.global_step, self.eval_trajs_dir)

        episode_reward = total_reward / episode
        with self.logger.log_and_dump_ctx(self.global_frame, ty='eval') as log:
            log('episode_reward', episode_reward)
            log('episode_length', step * self.cfg.action_repeat / episode)
            log('episode', self.global_episode)
            log('step', self.global_step)

        wandb.log({
            'eval/episode_reward': episode_reward,
            'eval/episode_length': step * self.cfg.action_repeat / episode,
            }, step=self.global_step)

        return episode_reward

    def train(self):
        # predicates
        train_until_step = utils.Until(self.cfg.num_train_frames,
                                       self.cfg.action_repeat)
        seed_until_step = utils.Until(self.cfg.num_seed_frames,
                                      self.cfg.action_repeat)
        eval_every_step = utils.Every(self.cfg.eval_every_frames,
                                      self.cfg.action_repeat)
        save_video_every_step = utils.Every(
            self.cfg.eval_every_frames * self.cfg.save_video_every_evals,
            self.cfg.action_repeat)

        episode_step, episode_reward = 0, 0
        time_step = self.train_env.reset()
        # self.replay_storage.add(time_step)
        self.replay_storage.add_my(time_step)
        self.train_video_recorder.init(time_step.observation)
        metrics = None
        # self.first_time_tmp = 1
        curr_observation = time_step.observation
        next_observation = time_step.observation
        first = True
        while train_until_step(self.global_step):
            if time_step.last():
                self._global_episode += 1
                self.train_video_recorder.save(f'{self.global_frame}.mp4')
                # wait until all the metrics schema is populated
                if metrics is not None:
                    # log stats
                    elapsed_time, total_time = self.timer.reset()
                    episode_frame = episode_step * self.cfg.action_repeat
                    with self.logger.log_and_dump_ctx(self.global_frame,
                                                        ty='train') as log:
                        log('fps', episode_frame / elapsed_time)
                        log('total_time', total_time)
                        log('episode_reward', episode_reward)
                        log('episode_length', episode_frame)
                        log('episode', self.global_episode)
                        log('buffer_size', len(self.replay_storage))
                        log('step', self.global_step)
                    wandb.log({
                        'train/fps': episode_frame / elapsed_time,
                        'train/total_time': total_time,
                        'train/episode_reward': episode_reward,
                        'train/episode_length': episode_frame,
                        }, step=self.global_step)

                # reset env
                time_step = self.train_env.reset()
                self.replay_storage.add(time_step)
                self.train_video_recorder.init(time_step.observation)
                # try to save snapshot
                if self.cfg.save_snapshot:
                    self.save_snapshot()
                episode_step = 0
                episode_reward = 0

            # try to evaluate
            if eval_every_step(self.global_step):
                self.logger.log('eval_total_time', self.timer.total_time(),
                                self.global_frame)

                # update best eval episode reward
                eval_episode_reward = self.eval(
                    save_video=save_video_every_step(self.global_step))
                if eval_episode_reward > self.best_episode_reward:
                    self._best_episode_reward = eval_episode_reward
                    # optionally save ckpt
                    if self.cfg.save_best_ckpt:
                        self.save_ckpt('best')

            # sample action
            with torch.no_grad(), utils.eval_mode(self.agent):
                action = self.agent.act(time_step.observation,
                                        self.global_step,
                                        eval_mode=False)

            # try to update the agent
            if not seed_until_step(self.global_step):
                metrics = self.agent.update(self.replay_iter, self.global_step)
                self.logger.log_metrics(metrics, self.global_frame, ty='train')
                wandb.log(metrics, step=self.global_step)

            # take env step
            time_step = self.train_env.step(action)
            episode_reward += time_step.reward
            self.replay_storage.add(time_step)
            self.train_video_recorder.record(time_step.observation)
            episode_step += 1
            self._global_step += 1

    def save_snapshot(self):
        snapshot = self.work_dir / 'snapshot.pt'
        keys_to_save = ['agent', 'timer', '_global_step', '_global_episode']
        payload = {k: self.__dict__[k] for k in keys_to_save}
        with snapshot.open('wb') as f:
            torch.save(payload, f)

    def load_snapshot(self):
        snapshot = self.work_dir / 'snapshot.pt'
        with snapshot.open('rb') as f:
            payload = torch.load(f)
        for k, v in payload.items():
            self.__dict__[k] = v

    def save_ckpt(self, save_fn):
        save_path = self.work_dir / f'{save_fn}.pt'
        keys_to_save = [
            'agent', 'timer', '_global_step', '_global_episode',
            '_best_episode_reward']
        ckpt = {k: self.__dict__[k] for k in keys_to_save}
        with save_path.open('wb') as f:
            torch.save(ckpt, f)

    def load_ckpt(self, load_path):
        load_path = Path(load_path)
        assert load_path.is_file(), f'[ckpt] {load_path} not found'
        with load_path.open('rb') as f:
            ckpt = torch.load(f, map_location='cpu')
        for k, v in ckpt.items():
            self.__dict__[k] = v


@hydra.main(config_path='cfgs', config_name='config')
def main(cfg):
    from train import Workspace as W
    root_dir = Path.cwd()
    workspace = W(cfg)
    snapshot = root_dir / 'snapshot.pt'
    if snapshot.exists():
        print(f'resuming: {snapshot}')
        workspace.load_snapshot()

    cfg_dict = utils.omegaconf_to_dict(workspace.cfg)
    wandb.init(
        project="mimex-dmc", name=str(workspace.work_dir).split('/')[-1],
        config=cfg_dict, mode=cfg.wandb_mode
        )
    workspace.train()
    wandb.finish()


if __name__ == '__main__':
    main()