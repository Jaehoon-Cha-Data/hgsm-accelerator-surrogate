#!/usr/bin/env python3

import os
import json
import copy
import random
from dataclasses import dataclass
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.analytic import LogExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.models.transforms.outcome import Standardize
from botorch.models.transforms.input import Normalize
from gpytorch.kernels import ScaleKernel, MaternKernel
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
import gpytorch
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.constraints import GreaterThan
from gpytorch.constraints import Interval
from model_single import SingleNet
from single_runner import ModelRunner, PARAM_NAMES, LOW, HIGH
DTYPE = torch.double
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LOW = np.asarray(LOW, dtype=np.float32)
HIGH = np.asarray(HIGH, dtype=np.float32)
DIM = len(PARAM_NAMES)
BO_N_INIT = 36
BO_ITERS = 60
BO_N_INIT_TRANSFER = 24
BO_ITERS_TRANSFER = 30
TRUST_REGION_SHRINK = 0.5
N_EXPERIMENT_RUNS = 10
SEED0 = 42
ACQ_NUM_RESTARTS = 40
ACQ_RAW_SAMPLES = 2048
S1_MODEL_PATH = os.path.join('shifted_systems_twiss_guided/bundles', 'S1_bundle.pt')
TARGET_MODEL_PATHS = {'S2': os.path.join('shifted_systems_twiss_guided/bundles', 'S2_bundle.pt'), 'S3': os.path.join('shifted_systems_twiss_guided/bundles', 'S3_bundle.pt'), 'S4': os.path.join('shifted_systems_twiss_guided/bundles', 'S4_bundle.pt'), 'S5': os.path.join('shifted_systems_twiss_guided/bundles', 'S5_bundle.pt'), 'S6': os.path.join('shifted_systems_twiss_guided/bundles', 'S6_bundle.pt'), 'S7': os.path.join('shifted_systems_twiss_guided/bundles', 'S7_bundle.pt'), 'S8': os.path.join('shifted_systems_twiss_guided/bundles', 'S8_bundle.pt'), 'S9': os.path.join('shifted_systems_twiss_guided/bundles', 'S9_bundle.pt'), 'S10': os.path.join('shifted_systems_twiss_guided/bundles', 'S10_bundle.pt')}
OUTDIR = f'transfer_study_results_BO_reordered_S1_to_S2_S10_tr_{TRUST_REGION_SHRINK}'
os.makedirs(OUTDIR, exist_ok=True)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def clip_to_bounds(x: np.ndarray, low=LOW, high=HIGH):
    return np.clip(x, low, high).astype(np.float32)

def cumulative_best(y_hist):
    y_hist = np.asarray(y_hist, dtype=float)
    return np.minimum.accumulate(y_hist)

def tighten_bounds(center: np.ndarray, low=LOW, high=HIGH, shrink=0.5):
    center = np.asarray(center, dtype=np.float32)
    full_width = high - low
    half_width = 0.5 * shrink * full_width
    new_low = np.maximum(low, center - half_width)
    new_high = np.minimum(high, center + half_width)
    return (new_low.astype(np.float32), new_high.astype(np.float32))

def save_json(obj, path):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)

def make_runner(model_path):
    return ModelRunner(model_path, SingleNet, key_p=1.0)

def mean_std_best(values):
    values = np.asarray(values, dtype=float)
    return {'mean': float(values.mean()), 'std': float(values.std(ddof=1)) if len(values) > 1 else 0.0, 'best': float(values.min()), 'all': values.tolist()}

def tensor_to_cpu_clone(x):
    if torch.is_tensor(x):
        return x.detach().cpu().clone()
    return copy.deepcopy(x)

@dataclass
class BOResult:
    best_x: list
    best_obj: float
    history_x: list
    history_y: list
    gp_final_state: dict | None = None
    gp_best_state: dict | None = None

def eval_batch_np(runner, X_np):
    ys = []
    for x in X_np:
        y = float(runner.evaluate(x.astype(np.float32)))
        ys.append(y)
    return np.asarray(ys, dtype=np.float64)[:, None]

def init_sobol_points(low, high, n, seed):
    d = len(low)
    sobol = torch.quasirandom.SobolEngine(dimension=d, scramble=True, seed=seed)
    U = sobol.draw(n).to(dtype=DTYPE, device=DEVICE)
    low_t = torch.tensor(low, dtype=DTYPE, device=DEVICE)
    high_t = torch.tensor(high, dtype=DTYPE, device=DEVICE)
    X = low_t + (high_t - low_t) * U
    return X

def extract_gp_transfer_state(gp):
    return {'lengthscale': gp.covar_module.base_kernel.lengthscale.detach().clone(), 'outputscale': gp.covar_module.outputscale.detach().clone()}

def apply_gp_warmstart(gp, state):
    with torch.no_grad():
        gp.covar_module.base_kernel.lengthscale.copy_(state['lengthscale'].to(gp.covar_module.base_kernel.lengthscale))
        gp.covar_module.outputscale.copy_(state['outputscale'].to(gp.covar_module.outputscale))

def fit_gp(X, Y, transfer_state=None):
    d = X.shape[-1]
    likelihood = GaussianLikelihood(noise_constraint=Interval(1e-06, 0.01))
    gp = SingleTaskGP(X, Y, likelihood=likelihood, covar_module=ScaleKernel(MaternKernel(nu=1.5, ard_num_dims=d, lengthscale_constraint=Interval(0.0001, 100.0))), outcome_transform=Standardize(m=1), input_transform=Normalize(d=d))
    if transfer_state is not None:
        apply_gp_warmstart(gp, transfer_state)
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    try:
        with gpytorch.settings.cholesky_max_tries(15), gpytorch.settings.cholesky_jitter(0.001):
            fit_gpytorch_mll(mll)
    except Exception as e:
        print(f'\n[Warning] L-BFGS-B failed. Falling back to Adam.')
        mll.train()
        optimizer = torch.optim.Adam(gp.parameters(), lr=0.05)
        train_inputs = gp.train_inputs[0]
        train_targets = gp.train_targets
        for _ in range(100):
            optimizer.zero_grad()
            with gpytorch.settings.cholesky_max_tries(15), gpytorch.settings.cholesky_jitter(0.001):
                output = gp(train_inputs)
                loss = -mll(output, train_targets).sum()
                loss.backward()
            optimizer.step()
        mll.eval()
    return gp

def run_bo(runner, low=LOW, high=HIGH, n_init=BO_N_INIT, iters=BO_ITERS, seed=42, warmstart_transfer_state=None):
    """
    If warmstart_kernel_hypers is None:
        scratch GP fit every time
    Else:
        warm-start kernel hyperparameters at every GP refit
    """
    set_seed(seed)
    X = init_sobol_points(low, high, n_init, seed)
    Y = torch.tensor(eval_batch_np(runner, X.detach().cpu().numpy()), dtype=DTYPE, device=DEVICE)
    gp = fit_gp(X, Y, transfer_state=warmstart_transfer_state)
    best_idx = int(torch.argmin(Y))
    best_obj = float(Y[best_idx].item())
    best_x = X[best_idx].detach().cpu().numpy().astype(np.float32)
    history_x = [row.tolist() for row in X.detach().cpu().numpy()]
    history_y = [float(v) for v in Y.detach().cpu().numpy().reshape(-1)]
    gp_best_state = extract_gp_transfer_state(gp)
    search_bounds = torch.tensor(np.vstack([low, high]), dtype=DTYPE, device=DEVICE)
    pbar = tqdm(range(iters), desc='BO', leave=True)
    for it in pbar:
        acq = LogExpectedImprovement(model=gp, best_f=Y.min(), maximize=False)
        cand, _ = optimize_acqf(acq_function=acq, bounds=search_bounds, q=1, num_restarts=ACQ_NUM_RESTARTS, raw_samples=ACQ_RAW_SAMPLES)
        distances = torch.cdist(cand, X)
        min_dist = distances.min().item()
        if min_dist < 0.0001:
            cand = init_sobol_points(low, high, 1, seed=it * 999)
        y_cand_np = eval_batch_np(runner, cand.detach().cpu().numpy())
        y_cand = torch.tensor(y_cand_np, dtype=DTYPE, device=DEVICE)
        X = torch.cat([X, cand], dim=0)
        Y = torch.cat([Y, y_cand], dim=0)
        gp = fit_gp(X, Y, transfer_state=warmstart_transfer_state)
        x_val = cand[0].detach().cpu().numpy().astype(np.float32)
        y_val = float(y_cand.item())
        history_x.append(x_val.tolist())
        history_y.append(y_val)
        if y_val < best_obj:
            best_obj = y_val
            best_x = x_val.copy()
            gp_best_state = extract_gp_transfer_state(gp)
        pbar.set_postfix(best=f'{best_obj:.6f}')
    gp_final_state = extract_gp_transfer_state(gp)
    if gp_best_state is None:
        gp_best_state = gp_final_state
    result = BOResult(best_x=best_x.tolist(), best_obj=best_obj, history_x=history_x, history_y=history_y, gp_final_state=gp_final_state, gp_best_state=gp_best_state)
    return (result, gp_final_state)

def study1_bo(s1_runner, s2_runner, seed):
    """
    Study 1:
    - S1 scratch BO over full bounds
    - S2 scratch BO over full bounds
    """
    res1, s1_kernel_hypers = run_bo(s1_runner, low=LOW, high=HIGH, n_init=BO_N_INIT, iters=BO_ITERS, seed=seed, warmstart_transfer_state=None)
    res2, s2_kernel_hypers = run_bo(s2_runner, low=LOW, high=HIGH, n_init=BO_N_INIT, iters=BO_ITERS, seed=seed + 1000, warmstart_transfer_state=None)
    x1_star = np.asarray(res1.best_x, dtype=np.float32)
    f2_x1star = float(s2_runner.evaluate(x1_star))
    return {'S1_best': res1, 'S2_best': res2, 'f2_x1star': f2_x1star, 's1_gp_final_state': res1.gp_final_state, 's1_gp_best_state': res1.gp_best_state, 's2_gp_final_state': res2.gp_final_state}

def study2_bo(s2_runner, x1_star, seed, shrink):
    """
    Study 2:
    - scratch BO
    - trust-region bounds
    """
    tr_low, tr_high = tighten_bounds(np.asarray(x1_star, dtype=np.float32), low=LOW, high=HIGH, shrink=shrink)
    res_tr, _ = run_bo(s2_runner, low=tr_low, high=tr_high, n_init=BO_N_INIT_TRANSFER, iters=BO_ITERS_TRANSFER, seed=seed, warmstart_transfer_state=None)
    return {'trust_region_low': tr_low.tolist(), 'trust_region_high': tr_high.tolist(), 'S2_TR_best': res_tr}

def study3a_bo_transfer(s2_runner, s1_gp_state, seed):
    """
    Study 3A:
    - full bounds
    - warm-start kernel hyperparameters every GP refit
    """
    res_transfer, _ = run_bo(s2_runner, low=LOW, high=HIGH, n_init=BO_N_INIT_TRANSFER, iters=BO_ITERS_TRANSFER, seed=seed, warmstart_transfer_state=s1_gp_state)
    return {'S2_transfer_gp_best': res_transfer}

def study4_bo(s2_runner, x1_star, s1_gp_state, seed, shrink):
    """
    Study 4:
    - trust-region bounds
    - warm-start kernel hyperparameters every GP refit
    """
    tr_low, tr_high = tighten_bounds(np.asarray(x1_star, dtype=np.float32), low=LOW, high=HIGH, shrink=shrink)
    res_both, _ = run_bo(s2_runner, low=tr_low, high=tr_high, n_init=BO_N_INIT_TRANSFER, iters=BO_ITERS_TRANSFER, seed=seed, warmstart_transfer_state=s1_gp_state)
    return {'trust_region_low': tr_low.tolist(), 'trust_region_high': tr_high.tolist(), 'S2_TR_transfer_gp_best': res_both}

def run_source_s1_only():
    source_pool = []
    raw_source_runs = []
    for run_idx in range(N_EXPERIMENT_RUNS):
        seed = SEED0 + run_idx
        print('\n' + '=' * 80)
        print(f'[Phase A] S1 only | run={run_idx} | seed={seed}')
        print('=' * 80)
        s1_runner = make_runner(S1_MODEL_PATH)
        res1, _ = run_bo(s1_runner, low=LOW, high=HIGH, n_init=BO_N_INIT, iters=BO_ITERS, seed=seed, warmstart_transfer_state=None)
        raw_source_runs.append({'run_idx': run_idx, 'seed': seed, 'S1_best': res1})
        source_pool.append({'run_idx': run_idx, 'seed': seed, 'S1_best': summarise_bo_result(res1)})
    s1_best_objs = [float(r['S1_best'].best_obj) for r in raw_source_runs]
    best_source_idx = int(np.argmin(s1_best_objs))
    best_source = raw_source_runs[best_source_idx]
    x1_star_best = np.asarray(best_source['S1_best'].best_x, dtype=np.float32)
    s1_gp_final_state = best_source['S1_best'].gp_final_state
    selected_source = {'run_idx': int(best_source['run_idx']), 'seed': int(best_source['seed']), 'criterion': 'lowest S1 best objective', 'S1_best_obj': float(best_source['S1_best'].best_obj), 'S1_best_x': [float(v) for v in best_source['S1_best'].best_x], 'x1_star_best': x1_star_best, 's1_gp_final_state': s1_gp_final_state}
    source_transfer_package = {'source_model': 'S1', 'selection_rule': 'lowest S1 best objective', 'run_idx': int(best_source['run_idx']), 'seed': int(best_source['seed']), 's1_best_obj': float(best_source['S1_best'].best_obj), 'x1_star_best': [float(v) for v in x1_star_best]}
    torch.save(s1_gp_final_state, os.path.join(OUTDIR, 's1_gp_final_state.pt'))
    save_json(source_transfer_package, os.path.join(OUTDIR, 'source_transfer_package.json'))
    return (source_pool, selected_source)

def run_target_studies(target_name, target_model_path, selected_source):
    x1_star_best = selected_source['x1_star_best']
    s1_gp_final_state = selected_source['s1_gp_final_state']
    target_results = {'scratch_runs': [], 'study2_transfer_runs': [], 'study3_gp_transfer_runs': [], 'study4_transfer_runs': [], 'summary': None}
    for run_idx in range(N_EXPERIMENT_RUNS):
        base_seed = SEED0 + 1000 + run_idx
        print('\n' + '=' * 80)
        print(f'[Phase B] {target_name} | run={run_idx} | base_seed={base_seed}')
        print('=' * 80)
        s_runner_scratch = make_runner(target_model_path)
        s_runner_2 = make_runner(target_model_path)
        s_runner_3 = make_runner(target_model_path)
        s_runner_4 = make_runner(target_model_path)
        f_target_x1star = float(s_runner_scratch.evaluate(x1_star_best))
        res_scratch, _ = run_bo(s_runner_scratch, low=LOW, high=HIGH, n_init=BO_N_INIT, iters=BO_ITERS, seed=base_seed, warmstart_transfer_state=None)
        print(f'\n[BO] Study 2 | shrink={TRUST_REGION_SHRINK}')
        bo2 = study2_bo(s_runner_2, x1_star_best, seed=base_seed + 10, shrink=TRUST_REGION_SHRINK)
        print('\n[BO] Study 3 GP')
        bo3_final = study3a_bo_transfer(s_runner_3, s1_gp_final_state, seed=base_seed + 23)
        print(f'\n[BO] Study 4 | shrink={TRUST_REGION_SHRINK}')
        bo4 = study4_bo(s_runner_4, x1_star_best, s1_gp_final_state, seed=base_seed + 30, shrink=TRUST_REGION_SHRINK)
        target_results['scratch_runs'].append({'run_idx': run_idx, 'seed': base_seed, 'target_best': summarise_bo_result(res_scratch), 'f_target_x1star': f_target_x1star})
        target_results['study2_transfer_runs'].append({'run_idx': run_idx, 'seed': base_seed + 10, 'trust_region_low': bo2['trust_region_low'], 'trust_region_high': bo2['trust_region_high'], 'S2_TR_best': summarise_bo_result(bo2['S2_TR_best'])})
        target_results['study3_gp_transfer_runs'].append({'run_idx': run_idx, 'seed': base_seed + 23, 'S2_transfer_gp_best': summarise_bo_result(bo3_final['S2_transfer_gp_best'])})
        target_results['study4_transfer_runs'].append({'run_idx': run_idx, 'seed': base_seed + 30, 'trust_region_low': bo4['trust_region_low'], 'trust_region_high': bo4['trust_region_high'], 'S2_TR_transfer_gp_best': summarise_bo_result(bo4['S2_TR_transfer_gp_best'])})
    scratch_vals = [r['target_best']['best_obj'] for r in target_results['scratch_runs']]
    f_x1_vals = [r['f_target_x1star'] for r in target_results['scratch_runs']]
    study2_vals = [r['S2_TR_best']['best_obj'] for r in target_results['study2_transfer_runs']]
    study3_vals = [r['S2_transfer_gp_best']['best_obj'] for r in target_results['study3_gp_transfer_runs']]
    study4_vals = [r['S2_TR_transfer_gp_best']['best_obj'] for r in target_results['study4_transfer_runs']]
    target_results['summary'] = {'scratch_target_best_obj': mean_std_best(scratch_vals), 'f_target_x1star': mean_std_best(f_x1_vals), 'S2_TR_best_obj': mean_std_best(study2_vals), 'S2_transfer_gp_final_best_obj': mean_std_best(study3_vals), 'S2_TR_transfer_gp_best_obj': mean_std_best(study4_vals)}
    return target_results

def summarise_bo_result(res: BOResult):
    return {'best_obj': float(res.best_obj), 'best_x': [float(v) for v in res.best_x], 'n_evals': len(res.history_y), 'best_curve': cumulative_best(res.history_y).tolist(), 'history_y': [float(v) for v in res.history_y]}

def build_summary(all_results):
    source_s1 = [r['S1_best']['best_obj'] for r in all_results['source_selection_pool']]
    return {'source': {'S1_best_obj': mean_std_best(source_s1)}, 'targets': {target_name: target_block['summary'] for target_name, target_block in all_results['targets'].items()}}

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
    target_block = all_results['targets']['S2']
    scratch_curves = [r['target_best']['best_curve'] for r in target_block['scratch_runs']]
    tr_curves_05 = [r['S2_TR_best']['best_curve'] for r in target_block['study2_transfer_runs']]
    gp_final_curves = [r['S2_transfer_gp_best']['best_curve'] for r in target_block['study3_gp_transfer_runs']]
    both_curves_05 = [r['S2_TR_transfer_gp_best']['best_curve'] for r in target_block['study4_transfer_runs']]
    plt.figure(figsize=(7, 5), dpi=200)
    ax = plt.gca()
    plot_mean_anytime(scratch_curves, 'BO scratch', ax)
    plot_mean_anytime(tr_curves_05, 'BO trust region 0.5', ax)
    plot_mean_anytime(gp_final_curves, 'BO transfer GP final', ax)
    plot_mean_anytime(both_curves_05, 'BO trust region 0.5', ax)
    ax.set_xlabel('True system evaluations')
    ax.set_ylabel('Best objective so far')
    ax.set_title('BO anytime curves on S2, mean ± std')
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def run_all():
    all_results = {'config': {'BO_N_INIT': BO_N_INIT, 'BO_ITERS': BO_ITERS, 'BO_N_INIT_TRANSFER': BO_N_INIT_TRANSFER, 'BO_ITERS_TRANSFER': BO_ITERS_TRANSFER, 'TRUST_REGION_SHRINK': TRUST_REGION_SHRINK, 'N_EXPERIMENT_RUNS': N_EXPERIMENT_RUNS, 'SEED0': SEED0, 'ACQ_NUM_RESTARTS': ACQ_NUM_RESTARTS, 'ACQ_RAW_SAMPLES': ACQ_RAW_SAMPLES, 'selection_rule': 'best S1 objective among source-only runs', 'study1_seed_range': [SEED0, SEED0 + N_EXPERIMENT_RUNS - 1], 'transfer_seed_range': [SEED0 + 1000, SEED0 + 1000 + N_EXPERIMENT_RUNS - 1], 'S1_MODEL_PATH': S1_MODEL_PATH, 'TARGET_MODEL_PATHS': TARGET_MODEL_PATHS, 'study_definitions': {'source': 'S1 scratch BO only', 'scratch': 'target scratch BO over full bounds', 'study2': 'scratch GP fit every time, trust-region bounds', 'study3': 'warm-start GP hyperparameters every GP refit', 'study4': 'same as study3a plus trust-region bounds'}}, 'source_selection_pool': [], 'selected_source_run': None, 'targets': {}, 'summary': None}
    source_pool, selected_source = run_source_s1_only()
    all_results['source_selection_pool'] = source_pool
    all_results['selected_source_run'] = {'run_idx': selected_source['run_idx'], 'seed': selected_source['seed'], 'criterion': selected_source['criterion'], 'S1_best_obj': selected_source['S1_best_obj'], 'S1_best_x': selected_source['S1_best_x']}
    print('\n' + '=' * 80)
    print('[Selection] Best source run chosen from S1-only runs')
    print(f"run_idx={selected_source['run_idx']}, seed={selected_source['seed']}, S1_best_obj={selected_source['S1_best_obj']:.6f}")
    print('=' * 80)
    for target_name, target_model_path in TARGET_MODEL_PATHS.items():
        target_results = run_target_studies(target_name=target_name, target_model_path=target_model_path, selected_source=selected_source)
        all_results['targets'][target_name] = target_results
        save_json(all_results, os.path.join(OUTDIR, 'all_results.json'))
    all_results['summary'] = build_summary(all_results)
    save_json(all_results, os.path.join(OUTDIR, 'all_results.json'))
    print('\nSaved results to:', OUTDIR)
    print('\nSummary:')
    print(json.dumps(all_results['summary'], indent=2))
if __name__ == '__main__':
    run_all()
