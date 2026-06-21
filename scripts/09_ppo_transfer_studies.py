#!/usr/bin/env python3

"""
PPO-based optimisation transfer studies with reordered experiment flow.

New experiment order
--------------------
Phase A: Study 1 only
    - Run S1 optimisation from scratch for N seeds
    - Run S2 optimisation from scratch for N seeds
    - For each Study 1 run, also evaluate f2(x1*)

Selection
    - Select the best S1 run among the Study 1 pool
      using the lowest S1 best objective

Phase B: Transfer studies with fixed source
    - Study 2: trust region only on S2, repeated for N seeds
    - Study 3B: PPO transfer only on S2, repeated for N seeds
    - Study 4: trust region + PPO transfer on S2, repeated for N seeds

Important PPO design
--------------------
- PPO action space is ALWAYS the same fixed global box
- Trust-region restriction is applied only inside env.step() by clipping
- PPO transfer is done by creating a fresh PPO model and copying policy weights
"""
from __future__ import annotations
import json
import os
import random
from dataclasses import dataclass, asdict
from typing import Callable, Iterable, List, Optional, Sequence, Tuple
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import torch
from model_single import SingleNet
from single_runner import ModelRunner
S1_MODEL_PATH = os.path.join('shifted_systems_twiss_guided/bundles', 'S1_bundle.pt')
TARGET_MODEL_PATHS = {'S2': os.path.join('shifted_systems_twiss_guided/bundles', 'S2_bundle.pt'), 'S3': os.path.join('shifted_systems_twiss_guided/bundles', 'S3_bundle.pt'), 'S4': os.path.join('shifted_systems_twiss_guided/bundles', 'S4_bundle.pt'), 'S5': os.path.join('shifted_systems_twiss_guided/bundles', 'S5_bundle.pt'), 'S6': os.path.join('shifted_systems_twiss_guided/bundles', 'S6_bundle.pt'), 'S7': os.path.join('shifted_systems_twiss_guided/bundles', 'S7_bundle.pt'), 'S8': os.path.join('shifted_systems_twiss_guided/bundles', 'S8_bundle.pt'), 'S9': os.path.join('shifted_systems_twiss_guided/bundles', 'S9_bundle.pt'), 'S10': os.path.join('shifted_systems_twiss_guided/bundles', 'S10_bundle.pt')}

def make_runner(model_path):
    runner_obj = ModelRunner(model_path, SingleNet, key_p=1.0)
    return lambda x: float(runner_obj.evaluate(np.asarray(x, dtype=np.float32)))
TRUST_REGION_SHRINK = 0.5
OUTDIR = f'transfer_study_results_RL_reordered_S1_to_S2_S10_tr_{TRUST_REGION_SHRINK}'
GLOBAL_LOW = np.asarray([-0.02431604, -0.72431606, -1.2743161, -0.574316, 12.025684, -23.474316, 12.225684, -9.674316, 3.0256839, -15.774316, 18.025684, -14.274316], dtype=np.float32)
GLOBAL_HIGH = np.asarray([3.924316, 3.2243161, 2.674316, 3.374316, 15.974316, -19.525684, 16.174316, -5.725684, 6.974316, -11.825684, 21.974316, -10.325684], dtype=np.float32)
PPO_TIMESTEPS = 150000
PPO_TIMESTEPS_TRANSFER = 75000
N_RESTARTS = 10
SEED0 = 42

@dataclass
class PPOResult:
    best_x: Optional[List[float]]
    best_obj: float
    history_x: List[List[float]]
    history_y: List[float]

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def save_json(obj, path: str) -> None:
    ensure_dir(os.path.dirname(path) or '.')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2)

def cumulative_best(y_hist):
    y_hist = np.asarray(y_hist, dtype=float)
    return np.minimum.accumulate(y_hist)

def mean_std_best(values):
    values = np.asarray(values, dtype=float)
    return {'mean': float(values.mean()), 'std': float(values.std(ddof=1)) if len(values) > 1 else 0.0, 'best': float(values.min()), 'all': values.tolist()}

def tighten_bounds(center: np.ndarray, shrink: float, global_low: np.ndarray=GLOBAL_LOW, global_high: np.ndarray=GLOBAL_HIGH) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a trust region centred at center with width proportional
    to the full global search box.
    """
    center = np.asarray(center, dtype=np.float32)
    global_low = np.asarray(global_low, dtype=np.float32)
    global_high = np.asarray(global_high, dtype=np.float32)
    if not 0.0 < shrink <= 1.0:
        raise ValueError(f'shrink must be in (0, 1], got {shrink}')
    full_width = global_high - global_low
    half_width = 0.5 * shrink * full_width
    tr_low = np.maximum(center - half_width, global_low)
    tr_high = np.minimum(center + half_width, global_high)
    tr_low = np.minimum(tr_low, tr_high)
    tr_high = np.maximum(tr_high, tr_low)
    return (tr_low.astype(np.float32), tr_high.astype(np.float32))

class OneStepBanditEnv(gym.Env):
    """
    One-step optimisation environment.

    The policy outputs an action in the fixed global action space.
    If trust_low and trust_high are provided, the action is projected into
    that trust region before the runner is evaluated.
    """
    metadata = {}

    def __init__(self, runner: Callable[[np.ndarray], float], global_low: np.ndarray, global_high: np.ndarray, trust_low: Optional[np.ndarray]=None, trust_high: Optional[np.ndarray]=None):
        super().__init__()
        self.runner = runner
        self.global_low = np.asarray(global_low, dtype=np.float32)
        self.global_high = np.asarray(global_high, dtype=np.float32)
        self.trust_low = None if trust_low is None else np.asarray(trust_low, dtype=np.float32)
        self.trust_high = None if trust_high is None else np.asarray(trust_high, dtype=np.float32)
        self.action_space = spaces.Box(low=self.global_low, high=self.global_high, shape=self.global_low.shape, dtype=np.float32)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.best_obj = float('inf')
        self.best_x: Optional[np.ndarray] = None
        self.history_x: List[np.ndarray] = []
        self.history_y: List[float] = []

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        obs = np.zeros((1,), dtype=np.float32)
        info = {}
        return (obs, info)

    def step(self, action):
        x = np.asarray(action, dtype=np.float32).copy()
        x = np.clip(x, self.global_low, self.global_high)
        if self.trust_low is not None and self.trust_high is not None:
            x = np.clip(x, self.trust_low, self.trust_high)
        obj = float(self.runner(x))
        self.history_x.append(x.copy())
        self.history_y.append(obj)
        if obj < self.best_obj:
            self.best_obj = obj
            self.best_x = x.copy()
        reward = -obj
        terminated = True
        truncated = False
        obs = np.zeros((1,), dtype=np.float32)
        info = {'x_used': x.copy(), 'obj': obj, 'best_obj': self.best_obj}
        return (obs, reward, terminated, truncated, info)

class TqdmPPOCallback(BaseCallback):

    def __init__(self, total_timesteps: int, verbose: int=0):
        super().__init__(verbose)
        self.total_timesteps_expected = int(total_timesteps)
        self.pbar = None

    def _on_training_start(self) -> None:
        self.pbar = tqdm(total=self.total_timesteps_expected, desc='PPO', leave=True)

    def _on_step(self) -> bool:
        if self.pbar is not None:
            self.pbar.update(self.model.n_envs)
            try:
                best_obj = self.training_env.get_attr('best_obj')[0]
                if best_obj is not None and np.isfinite(best_obj):
                    self.pbar.set_postfix(best=f'{best_obj:.6f}')
            except Exception:
                pass
        return True

    def _on_training_end(self) -> None:
        if self.pbar is not None:
            remaining = self.total_timesteps_expected - self.pbar.n
            if remaining > 0:
                self.pbar.update(remaining)
            self.pbar.close()
            self.pbar = None

def build_ppo_agent(vec_env, seed: int) -> PPO:
    return PPO(policy='MlpPolicy', env=vec_env, seed=seed, verbose=0, device='cpu', n_steps=64, batch_size=64, gamma=0.0, learning_rate=0.0003, ent_coef=0.01)

def copy_ppo_weights(source_agent: PPO, target_agent: PPO) -> None:
    target_agent.policy.load_state_dict(source_agent.policy.state_dict())

def train_ppo(runner: Callable[[np.ndarray], float], total_timesteps: int, seed: int, global_low: np.ndarray=GLOBAL_LOW, global_high: np.ndarray=GLOBAL_HIGH, init_model: Optional[PPO]=None, transfer_weights: bool=False, trust_low: Optional[np.ndarray]=None, trust_high: Optional[np.ndarray]=None) -> Tuple[PPO, PPOResult]:
    """
    Train PPO on the one-step bandit env.

    Important:
    - action space is defined only by global_low / global_high
    - trust region is internal to the env and does not alter action_space
    """
    set_seed(seed)
    global_low = np.asarray(global_low, dtype=np.float32)
    global_high = np.asarray(global_high, dtype=np.float32)
    if trust_low is not None:
        trust_low = np.asarray(trust_low, dtype=np.float32)
    if trust_high is not None:
        trust_high = np.asarray(trust_high, dtype=np.float32)

    def make_env():

        def _init():
            env = OneStepBanditEnv(runner=runner, global_low=global_low, global_high=global_high, trust_low=trust_low, trust_high=trust_high)
            env.reset(seed=seed)
            return env
        return _init
    vec_env = DummyVecEnv([make_env()])
    agent = build_ppo_agent(vec_env, seed)
    if init_model is not None and transfer_weights:
        if init_model.observation_space != agent.observation_space:
            raise ValueError(f'Cannot transfer PPO weights because observation spaces differ:\nsource: {init_model.observation_space}\ntarget: {agent.observation_space}')
        if init_model.action_space != agent.action_space:
            raise ValueError('Cannot transfer PPO weights because action spaces differ.\nKeep global_low/global_high fixed across all PPO studies.')
        copy_ppo_weights(init_model, agent)
    callback = TqdmPPOCallback(total_timesteps=total_timesteps)
    agent.learn(total_timesteps=total_timesteps, callback=callback)
    env0 = vec_env.envs[0]
    result = PPOResult(best_x=env0.best_x.tolist() if env0.best_x is not None else None, best_obj=float(env0.best_obj), history_x=[x.tolist() for x in env0.history_x], history_y=[float(y) for y in env0.history_y])
    return (agent, result)

def study1_ppo(s1_runner: Callable[[np.ndarray], float], s2_runner: Callable[[np.ndarray], float], seed: int):
    """
    Study 1:
    Optimise S1 and S2 independently from scratch.
    Also evaluate f2(x1*).
    """
    agent_s1, res_s1 = train_ppo(runner=s1_runner, total_timesteps=PPO_TIMESTEPS, seed=seed, global_low=GLOBAL_LOW, global_high=GLOBAL_HIGH, init_model=None, transfer_weights=False, trust_low=None, trust_high=None)
    _, res_s2 = train_ppo(runner=s2_runner, total_timesteps=PPO_TIMESTEPS, seed=seed + 1000, global_low=GLOBAL_LOW, global_high=GLOBAL_HIGH, init_model=None, transfer_weights=False, trust_low=None, trust_high=None)
    if res_s1.best_x is None:
        raise RuntimeError('Study 1 on S1 did not produce best_x.')
    x1_star = np.asarray(res_s1.best_x, dtype=np.float32)
    f2_x1star = float(s2_runner(x1_star))
    return {'ppo_agent_s1': agent_s1, 'S1_best': res_s1, 'S2_best': res_s2, 'f2_x1star': f2_x1star}

def study2_ppo(s2_runner: Callable[[np.ndarray], float], x1_star: Sequence[float], seed: int):
    """
    Study 2:
    No transfer. S2 is optimised inside the trust region around x1*.
    """
    tr_low, tr_high = tighten_bounds(np.asarray(x1_star, dtype=np.float32), shrink=TRUST_REGION_SHRINK, global_low=GLOBAL_LOW, global_high=GLOBAL_HIGH)
    _, res_tr = train_ppo(runner=s2_runner, total_timesteps=PPO_TIMESTEPS_TRANSFER, seed=seed, global_low=GLOBAL_LOW, global_high=GLOBAL_HIGH, init_model=None, transfer_weights=False, trust_low=tr_low, trust_high=tr_high)
    return {'trust_region_low': tr_low.tolist(), 'trust_region_high': tr_high.tolist(), 'S2_TR_best': res_tr}

def study3_ppo_transfer(s2_runner: Callable[[np.ndarray], float], agent_s1: PPO, seed: int):
    """
    Study 3B:
    Transfer PPO weights from S1 to S2 on the full global box.
    """
    _, res_transfer = train_ppo(runner=s2_runner, total_timesteps=PPO_TIMESTEPS_TRANSFER, seed=seed, global_low=GLOBAL_LOW, global_high=GLOBAL_HIGH, init_model=agent_s1, transfer_weights=True, trust_low=None, trust_high=None)
    return {'S2_transfer_ppo_best': res_transfer}

def study4_ppo(s2_runner: Callable[[np.ndarray], float], x1_star: Sequence[float], agent_s1: PPO, seed: int):
    """
    Study 4:
    Transfer PPO weights from S1 to S2 and constrain actions
    internally to the trust region around x1*.
    """
    tr_low, tr_high = tighten_bounds(np.asarray(x1_star, dtype=np.float32), shrink=TRUST_REGION_SHRINK, global_low=GLOBAL_LOW, global_high=GLOBAL_HIGH)
    _, res_both = train_ppo(runner=s2_runner, total_timesteps=PPO_TIMESTEPS_TRANSFER, seed=seed, global_low=GLOBAL_LOW, global_high=GLOBAL_HIGH, init_model=agent_s1, transfer_weights=True, trust_low=tr_low, trust_high=tr_high)
    return {'trust_region_low': tr_low.tolist(), 'trust_region_high': tr_high.tolist(), 'S2_TR_transfer_ppo_best': res_both}

def run_source_s1_only(s1_runner: Callable[[np.ndarray], float], source_seeds: Sequence[int]):
    raw_source_runs = []
    source_pool = []
    for run_idx, seed in enumerate(source_seeds):
        print('\n' + '=' * 80)
        print(f'[Source] S1 only | run={run_idx} | seed={seed}')
        print('=' * 80)
        agent_s1, res_s1 = train_ppo(runner=s1_runner, total_timesteps=PPO_TIMESTEPS, seed=seed, global_low=GLOBAL_LOW, global_high=GLOBAL_HIGH, init_model=None, transfer_weights=False, trust_low=None, trust_high=None)
        raw_source_runs.append({'run_idx': run_idx, 'seed': seed, 'agent_s1': agent_s1, 'res_s1': res_s1})
        source_pool.append({'run_idx': run_idx, 'seed': seed, 'S1_best': summarise_ppo_result(res_s1)})
    s1_best_objs = [float(item['res_s1'].best_obj) for item in raw_source_runs]
    best_source_idx = int(np.argmin(s1_best_objs))
    best_source = raw_source_runs[best_source_idx]
    if best_source['res_s1'].best_x is None:
        raise RuntimeError('Selected S1 source run has no valid best_x.')
    model_save_path = os.path.join(OUTDIR, 'selected_source_agent_s1.zip')
    best_source['agent_s1'].save(model_save_path)
    print(f'[Saved] Selected S1 PPO model -> {model_save_path}')
    x1_star_best = np.asarray(best_source['res_s1'].best_x, dtype=np.float32)
    source_transfer_package = {'source_model': 'S1', 'selection_rule': 'lowest S1 best objective', 'run_idx': int(best_source['run_idx']), 'seed': int(best_source['seed']), 'S1_best_obj': float(best_source['res_s1'].best_obj), 'S1_best_x': [float(v) for v in x1_star_best], 'ppo_model_path': model_save_path}
    save_json(source_transfer_package, os.path.join(OUTDIR, 'source_transfer_package_ppo.json'))
    selected_source = {'run_idx': int(best_source['run_idx']), 'seed': int(best_source['seed']), 'S1_best_obj': float(best_source['res_s1'].best_obj), 'S1_best_x': [float(v) for v in x1_star_best], 'agent_s1': best_source['agent_s1'], 'agent_s1_path': model_save_path, 'x1_star': x1_star_best}
    return {'source_pool': source_pool, 'selected_source': selected_source}

def run_target_transfer_studies(target_name: str, target_model_path: str, selected_source: dict, transfer_seeds: Sequence[int]):
    x1_star = selected_source['x1_star']
    agent_s1 = selected_source['agent_s1']
    target_results = {'target_name': target_name, 'scratch_runs': [], 'study2_transfer_runs': [], 'study3_transfer_runs': [], 'study4_transfer_runs': [], 'summary': None}
    for run_idx, seed in enumerate(transfer_seeds):
        print('\n' + '=' * 80)
        print(f'[Target {target_name}] run={run_idx} | base_seed={seed}')
        print('=' * 80)
        runner_scratch = make_runner(target_model_path)
        runner_eval = make_runner(target_model_path)
        runner_2 = make_runner(target_model_path)
        runner_3 = make_runner(target_model_path)
        runner_4 = make_runner(target_model_path)
        _, res_scratch = train_ppo(runner=runner_scratch, total_timesteps=PPO_TIMESTEPS, seed=seed, global_low=GLOBAL_LOW, global_high=GLOBAL_HIGH, init_model=None, transfer_weights=False, trust_low=None, trust_high=None)
        f_target_x1star = float(runner_eval(x1_star))
        ppo2 = study2_ppo(runner_2, x1_star, seed=seed + 10)
        ppo3 = study3_ppo_transfer(runner_3, agent_s1, seed=seed + 20)
        ppo4 = study4_ppo(runner_4, x1_star, agent_s1, seed=seed + 30)
        target_results['scratch_runs'].append({'run_idx': run_idx, 'seed': seed, 'target_best': summarise_ppo_result(res_scratch), 'f_target_x1star': f_target_x1star})
        target_results['study2_transfer_runs'].append({'run_idx': run_idx, 'seed': seed + 10, 'trust_region_low': ppo2['trust_region_low'], 'trust_region_high': ppo2['trust_region_high'], 'S2_TR_best': summarise_ppo_result(ppo2['S2_TR_best'])})
        target_results['study3_transfer_runs'].append({'run_idx': run_idx, 'seed': seed + 20, 'S2_transfer_ppo_best': summarise_ppo_result(ppo3['S2_transfer_ppo_best'])})
        target_results['study4_transfer_runs'].append({'run_idx': run_idx, 'seed': seed + 30, 'trust_region_low': ppo4['trust_region_low'], 'trust_region_high': ppo4['trust_region_high'], 'S2_TR_transfer_ppo_best': summarise_ppo_result(ppo4['S2_TR_transfer_ppo_best'])})
    scratch_vals = [r['target_best']['best_obj'] for r in target_results['scratch_runs']]
    f_x1_vals = [r['f_target_x1star'] for r in target_results['scratch_runs']]
    study2_vals = [r['S2_TR_best']['best_obj'] for r in target_results['study2_transfer_runs']]
    study3_vals = [r['S2_transfer_ppo_best']['best_obj'] for r in target_results['study3_transfer_runs']]
    study4_vals = [r['S2_TR_transfer_ppo_best']['best_obj'] for r in target_results['study4_transfer_runs']]
    target_results['summary'] = {'scratch_target_best_obj': mean_std_best(scratch_vals), 'f_target_x1star': mean_std_best(f_x1_vals), 'trust_region_best_obj': mean_std_best(study2_vals), 'transfer_best_obj': mean_std_best(study3_vals), 'both_best_obj': mean_std_best(study4_vals)}
    return target_results

def result_to_jsonable(obj):
    if isinstance(obj, PPOResult):
        return asdict(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    return obj

def summarise_ppo_result(res: PPOResult):
    return {'best_obj': float(res.best_obj), 'best_x': None if res.best_x is None else [float(v) for v in res.best_x], 'n_evals': len(res.history_y), 'best_curve': cumulative_best(res.history_y).tolist(), 'history_y': [float(v) for v in res.history_y]}

def build_summary(all_results):
    study1_s1 = [r['S1_best']['best_obj'] for r in all_results['study1_selection_pool']]
    study1_s2 = [r['S2_best']['best_obj'] for r in all_results['study1_selection_pool']]
    study1_f2x1 = [r['f2_x1star'] for r in all_results['study1_selection_pool']]
    study2_vals = [r['S2_TR_best']['best_obj'] for r in all_results['study2_transfer_runs']]
    study3_vals = [r['S2_transfer_ppo_best']['best_obj'] for r in all_results['study3_transfer_runs']]
    study4_vals = [r['S2_TR_transfer_ppo_best']['best_obj'] for r in all_results['study4_transfer_runs']]
    return {'study1': {'S1_best_obj': mean_std_best(study1_s1), 'S2_best_obj': mean_std_best(study1_s2), 'f2_x1star': mean_std_best(study1_f2x1)}, 'study2': {'S2_TR_best_obj': mean_std_best(study2_vals)}, 'study3': {'S2_transfer_ppo_best_obj': mean_std_best(study3_vals)}, 'study4': {'S2_TR_transfer_ppo_best_obj': mean_std_best(study4_vals)}}

def plot_anytime(curves, labels, title, save_path):
    plt.figure(figsize=(7, 5), dpi=200)
    for c, label in zip(curves, labels):
        plt.plot(c, label=label)
    plt.xlabel('True system evaluations')
    plt.ylabel('Best objective so far')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_mean_anytime(curve_list, label, ax):
    min_len = min((len(c) for c in curve_list))
    arr = np.array([np.asarray(c[:min_len], dtype=float) for c in curve_list], dtype=float)
    mean_curve = arr.mean(axis=0)
    std_curve = arr.std(axis=0, ddof=1) if arr.shape[0] > 1 else np.zeros_like(mean_curve)
    xs = np.arange(min_len)
    ax.plot(xs, mean_curve, label=label)
    ax.fill_between(xs, mean_curve - std_curve, mean_curve + std_curve, alpha=0.2)

def plot_mean_anytime_comparison(all_results, save_path):
    scratch_curves = [r['S2_best']['best_curve'] for r in all_results['study1_selection_pool']]
    tr_curves = [r['S2_TR_best']['best_curve'] for r in all_results['study2_transfer_runs']]
    transfer_curves = [r['S2_transfer_ppo_best']['best_curve'] for r in all_results['study3_transfer_runs']]
    both_curves = [r['S2_TR_transfer_ppo_best']['best_curve'] for r in all_results['study4_transfer_runs']]
    plt.figure(figsize=(7, 5), dpi=200)
    ax = plt.gca()
    plot_mean_anytime(scratch_curves, 'PPO scratch', ax)
    plot_mean_anytime(tr_curves, 'PPO trust region', ax)
    plot_mean_anytime(transfer_curves, 'PPO transfer', ax)
    plot_mean_anytime(both_curves, 'PPO both', ax)
    ax.set_xlabel('True system evaluations')
    ax.set_ylabel('Best objective so far')
    ax.set_title('PPO anytime curves on S2, mean ± std')
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def run_all(runner_pairs: Iterable[Tuple[Callable[[np.ndarray], float], Callable[[np.ndarray], float]]], study1_seeds: Optional[Sequence[int]]=None, transfer_seeds: Optional[Sequence[int]]=None):
    """
    Reordered PPO experiments.

    Phase A
    -------
    Run Study 1 many times and select the best S1 source run.

    Phase B
    -------
    Using the fixed selected source from Phase A, run:
        - Study 2 many times
        - Study 3B many times
        - Study 4 many times

    Parameters
    ----------
    runner_pairs:
        iterable of (s1_runner, s2_runner) pairs for Study 1 pool

    study1_seeds:
        seeds for Study 1 runs
        if None, uses [SEED0, ..., SEED0 + N_RESTARTS - 1]

    transfer_seeds:
        seeds for transfer runs
        if None, uses [SEED0 + 1000, ..., SEED0 + 1000 + N_RESTARTS - 1]
    """
    ensure_dir(OUTDIR)
    runner_pairs = list(runner_pairs)
    if len(runner_pairs) == 0:
        raise ValueError('runner_pairs is empty. Provide at least one (s1_runner, s2_runner) pair.')
    if study1_seeds is None:
        study1_seeds = [SEED0 + i for i in range(N_RESTARTS)]
    else:
        study1_seeds = list(study1_seeds)
    if transfer_seeds is None:
        transfer_seeds = [SEED0 + 1000 + i for i in range(N_RESTARTS)]
    else:
        transfer_seeds = list(transfer_seeds)
    if len(runner_pairs) == 1 and len(study1_seeds) > 1:
        runner_pairs = runner_pairs * len(study1_seeds)
    if len(runner_pairs) != len(study1_seeds):
        raise ValueError(f'Number of Study 1 seeds ({len(study1_seeds)}) must match number of runner_pairs ({len(runner_pairs)}), unless a single runner pair is provided.')
    all_results = {'config': {'PPO_TIMESTEPS': PPO_TIMESTEPS, 'PPO_TIMESTEPS_TRANSFER': PPO_TIMESTEPS_TRANSFER, 'TRUST_REGION_SHRINK': TRUST_REGION_SHRINK, 'N_RESTARTS': N_RESTARTS, 'SEED0': SEED0, 'selection_rule': 'best S1 objective among Study 1 runs', 'GLOBAL_LOW': GLOBAL_LOW.tolist(), 'GLOBAL_HIGH': GLOBAL_HIGH.tolist()}, 'study1_selection_pool': [], 'selected_source_run': None, 'study2_transfer_runs': [], 'study3_transfer_runs': [], 'study4_transfer_runs': [], 'summary': None}
    raw_study1 = []
    for run_idx, ((s1_runner, s2_runner), seed) in enumerate(zip(runner_pairs, study1_seeds)):
        print('\n' + '=' * 80)
        print(f'[Phase A] Study 1 | run={run_idx} | seed={seed}')
        print('=' * 80)
        ppo1 = study1_ppo(s1_runner, s2_runner, seed)
        raw_study1.append({'run_idx': run_idx, 'seed': seed, 's1_runner': s1_runner, 's2_runner': s2_runner, 'ppo1': ppo1})
        all_results['study1_selection_pool'].append({'run_idx': run_idx, 'seed': seed, 'S1_best': summarise_ppo_result(ppo1['S1_best']), 'S2_best': summarise_ppo_result(ppo1['S2_best']), 'f2_x1star': float(ppo1['f2_x1star'])})
        save_json(all_results, os.path.join(OUTDIR, 'all_results_ppo.json'))
    s1_best_objs = [float(item['ppo1']['S1_best'].best_obj) for item in raw_study1]
    best_source_idx = int(np.argmin(s1_best_objs))
    best_source = raw_study1[best_source_idx]
    best_source_run_idx = int(best_source['run_idx'])
    best_source_seed = int(best_source['seed'])
    best_source_ppo1 = best_source['ppo1']
    best_s2_runner = best_source['s2_runner']
    if best_source_ppo1['S1_best'].best_x is None:
        raise RuntimeError('Selected Study 1 source run has no valid best_x.')
    x1_star_best = np.asarray(best_source_ppo1['S1_best'].best_x, dtype=np.float32)
    agent_s1_best = best_source_ppo1['ppo_agent_s1']
    all_results['selected_source_run'] = {'run_idx': best_source_run_idx, 'seed': best_source_seed, 'criterion': 'lowest S1 best objective', 'S1_best_obj': float(best_source_ppo1['S1_best'].best_obj), 'S1_best_x': [float(v) for v in best_source_ppo1['S1_best'].best_x], 'f2_x1star': float(best_source_ppo1['f2_x1star'])}
    print('\n' + '=' * 80)
    print('[Selection] Best source run chosen from Study 1')
    print(f"run_idx={best_source_run_idx}, seed={best_source_seed}, S1_best_obj={best_source_ppo1['S1_best'].best_obj:.6f}")
    print('=' * 80)
    save_json(all_results, os.path.join(OUTDIR, 'all_results_ppo.json'))
    for run_idx, seed in enumerate(transfer_seeds):
        print('\n' + '=' * 80)
        print(f'[Phase B] Transfer runs | run={run_idx} | base_seed={seed}')
        print('=' * 80)
        print('\n[PPO] Study 2')
        ppo2 = study2_ppo(best_s2_runner, x1_star_best, seed=seed + 10)
        print('\n[PPO] Study 3B')
        ppo3 = study3_ppo_transfer(best_s2_runner, agent_s1_best, seed=seed + 20)
        print('\n[PPO] Study 4')
        ppo4 = study4_ppo(best_s2_runner, x1_star_best, agent_s1_best, seed=seed + 30)
        all_results['study2_transfer_runs'].append({'run_idx': run_idx, 'seed': seed + 10, 'trust_region_low': ppo2['trust_region_low'], 'trust_region_high': ppo2['trust_region_high'], 'S2_TR_best': summarise_ppo_result(ppo2['S2_TR_best'])})
        all_results['study3_transfer_runs'].append({'run_idx': run_idx, 'seed': seed + 20, 'S2_transfer_ppo_best': summarise_ppo_result(ppo3['S2_transfer_ppo_best'])})
        all_results['study4_transfer_runs'].append({'run_idx': run_idx, 'seed': seed + 30, 'trust_region_low': ppo4['trust_region_low'], 'trust_region_high': ppo4['trust_region_high'], 'S2_TR_transfer_ppo_best': summarise_ppo_result(ppo4['S2_TR_transfer_ppo_best'])})
        save_json(all_results, os.path.join(OUTDIR, 'all_results_ppo.json'))
    all_results['summary'] = build_summary(all_results)
    save_json(all_results, os.path.join(OUTDIR, 'all_results_ppo.json'))
    if len(all_results['study1_selection_pool']) > 0 and len(all_results['study2_transfer_runs']) > 0 and (len(all_results['study3_transfer_runs']) > 0) and (len(all_results['study4_transfer_runs']) > 0):
        plot_anytime(curves=[all_results['study1_selection_pool'][0]['S2_best']['best_curve'], all_results['study2_transfer_runs'][0]['S2_TR_best']['best_curve'], all_results['study3_transfer_runs'][0]['S2_transfer_ppo_best']['best_curve'], all_results['study4_transfer_runs'][0]['S2_TR_transfer_ppo_best']['best_curve']], labels=['PPO scratch', 'PPO trust region', 'PPO transfer', 'PPO both'], title='PPO anytime curves on S2, example run', save_path=os.path.join(OUTDIR, 'ppo_anytime_example_run0.png'))
        plot_mean_anytime_comparison(all_results, save_path=os.path.join(OUTDIR, 'ppo_anytime_mean_std.png'))
    return all_results

def run_all_targets(s1_runner: Callable[[np.ndarray], float], target_model_paths: dict[str, str], source_seeds: Optional[Sequence[int]]=None, transfer_seeds: Optional[Sequence[int]]=None):
    ensure_dir(OUTDIR)
    if source_seeds is None:
        source_seeds = [SEED0 + i for i in range(N_RESTARTS)]
    else:
        source_seeds = list(source_seeds)
    if transfer_seeds is None:
        transfer_seeds = [SEED0 + 1000 + i for i in range(N_RESTARTS)]
    else:
        transfer_seeds = list(transfer_seeds)
    source_info = run_source_s1_only(s1_runner, source_seeds)
    selected_source = source_info['selected_source']
    print('\n' + '=' * 80)
    print('[Selection] Fixed S1 source selected')
    print(f"run_idx={selected_source['run_idx']}, seed={selected_source['seed']}, S1_best_obj={selected_source['S1_best_obj']:.6f}")
    print('=' * 80)
    all_results = {'config': {'PPO_TIMESTEPS': PPO_TIMESTEPS, 'PPO_TIMESTEPS_TRANSFER': PPO_TIMESTEPS_TRANSFER, 'TRUST_REGION_SHRINK': TRUST_REGION_SHRINK, 'N_RESTARTS': N_RESTARTS, 'SEED0': SEED0, 'source_model': 'S1', 'target_models': list(target_model_paths.keys())}, 'source_pool': source_info['source_pool'], 'selected_source': {'run_idx': selected_source['run_idx'], 'seed': selected_source['seed'], 'S1_best_obj': selected_source['S1_best_obj'], 'S1_best_x': selected_source['S1_best_x']}, 'targets': {}}
    for target_name, target_model_path in target_model_paths.items():
        target_results = run_target_transfer_studies(target_name=target_name, target_model_path=target_model_path, selected_source=selected_source, transfer_seeds=transfer_seeds)
        all_results['targets'][target_name] = target_results
        save_json(all_results, os.path.join(OUTDIR, 'all_results_ppo_multi_target.json'))
    return all_results
if __name__ == '__main__':
    s1_runner = make_runner(S1_MODEL_PATH)
    results = run_all_targets(s1_runner=s1_runner, target_model_paths=TARGET_MODEL_PATHS, source_seeds=[SEED0 + i for i in range(N_RESTARTS)], transfer_seeds=[SEED0 + 1000 + i for i in range(N_RESTARTS)])
    print('\nSaved PPO results to:', os.path.join(OUTDIR, 'all_results_ppo_multi_target.json'))
