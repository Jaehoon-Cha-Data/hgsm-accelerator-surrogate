#!/usr/bin/env python3

import os
import types
import random
import numpy as np
import torch
import h5py
import time
from torch.utils.data import DataLoader, TensorDataset
from models_second import Mixer, HypyerNet

def set_seed(seed: int=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def x_get_mean_std(data):
    return (data.mean(0), data.std(0))

def y_get_mean_std(data):
    means = np.mean(data, axis=(0, 2), keepdims=True)
    stds = np.std(data, axis=(0, 2), keepdims=True)
    return (means, stds)

def load_h5(path):
    with h5py.File(path, 'r') as h5:
        X = h5['inputs'][:].astype(np.float32)
        Y = h5['outputs'][:].astype(np.float32)
    return (X, Y)

def load_datasets(save_dir):
    x_train, y_train = load_h5(f'{save_dir}/sim_good.h5')
    x_mean, x_std = x_get_mean_std(x_train)
    x_train = (x_train - x_mean) / x_std
    y_mean, y_std = y_get_mean_std(y_train)
    y_train[:, 1:-1, :] = (y_train[:, 1:-1, :] - y_mean[:, 1:-1, :]) / y_std[:, 1:-1, :]
    y_train = np.transpose(y_train, (0, 2, 1))
    return (x_train, y_train, x_mean, x_std, y_mean, y_std)

def build_inputs_ft(T=200, embedding_dim=100, p=1.0):
    location = np.float32(np.linspace(0, 1.0, T, endpoint=False))

    def input_encoder(x, a, b):
        out = np.concatenate([a * np.sin(2.0 * np.pi * x[..., None] * b), a * np.cos(2.0 * np.pi * x[..., None] * b)], axis=-1)
        return out / np.linalg.norm(a)
    bvals = np.float32(np.arange(1, embedding_dim + 1))
    a = bvals ** (-np.float32(p))
    inputs_ft = input_encoder(location, a, bvals)
    return torch.from_numpy(inputs_ft[None, ...].astype(np.float32))

def unnormalize_y(pred, y_mean, y_std):
    """
    pred: (..., 12)
    Only channels 1:-1 were normalised during training.
    """
    pred_phys = pred.clone()
    y_mean_t = torch.as_tensor(y_mean, dtype=pred.dtype, device=pred.device)[0, :, 0]
    y_std_t = torch.as_tensor(y_std, dtype=pred.dtype, device=pred.device)[0, :, 0]
    pred_phys[..., 1:-1] = pred_phys[..., 1:-1] * y_std_t[1:-1] + y_mean_t[1:-1]
    return pred_phys

def get_twiss_x(outputs, eps=1e-10):
    """
    Channel mapping:
    outputs[..., 3]  = sigma_x
    outputs[..., 5]  = emit_x
    outputs[..., 9]  = cov_xxp
    """
    sigma_x = torch.clamp(outputs[..., 3], min=eps)
    emit_x = torch.clamp(outputs[..., 5], min=eps)
    cov_xxp = outputs[..., 9]
    beta_x = torch.clamp(sigma_x ** 2 / emit_x, min=eps)
    alpha_x = -cov_xxp / torch.clamp(sigma_x ** 2, min=eps)
    gamma_x = (1.0 + alpha_x ** 2) / beta_x
    return (alpha_x, beta_x, gamma_x)

def get_twiss_y(outputs, eps=1e-10):
    """
    Channel mapping:
    outputs[..., 4]  = sigma_y
    outputs[..., 6]  = emit_y
    outputs[..., 10] = cov_yyp
    """
    sigma_y = torch.clamp(outputs[..., 4], min=eps)
    emit_y = torch.clamp(outputs[..., 6], min=eps)
    cov_yyp = outputs[..., 10]
    beta_y = torch.clamp(sigma_y ** 2 / emit_y, min=eps)
    alpha_y = -cov_yyp / torch.clamp(sigma_y ** 2, min=eps)
    gamma_y = (1.0 + alpha_y ** 2) / beta_y
    return (alpha_y, beta_y, gamma_y)

def twiss_rms_mismatch_np(pred, target, y_mean, y_std, eps=0.0001):
    """
    pred, target: (..., 12) in normalised space.

    Returns raw betatron mismatch factors:
        Hx
        Hy
        H_mean = 0.5 * (Hx + Hy)
        excess_H_mean = 0.5 * [max(Hx - 1, 0) + max(Hy - 1, 0)]
    """
    pred_phys = unnormalize_y(pred, y_mean, y_std)
    target_phys = unnormalize_y(target, y_mean, y_std)
    ax_p, bx_p, gx_p = get_twiss_x(pred_phys, eps)
    ay_p, by_p, gy_p = get_twiss_y(pred_phys, eps)
    ax_t, bx_t, gx_t = get_twiss_x(target_phys, eps)
    ay_t, by_t, gy_t = get_twiss_y(target_phys, eps)
    Hx = 0.5 * (bx_t * gx_p - 2.0 * ax_t * ax_p + gx_t * bx_p)
    Hy = 0.5 * (by_t * gy_p - 2.0 * ay_t * ay_p + gy_t * by_p)
    H_mean = 0.5 * (Hx + Hy)
    excess_Hx = torch.clamp(Hx - 1.0, min=0.0)
    excess_Hy = torch.clamp(Hy - 1.0, min=0.0)
    excess_H_mean = 0.5 * (excess_Hx + excess_Hy)
    return H_mean

class HyperCalibrator:

    def __init__(self, model_params, weights_folder, y_mean, y_std, device=None, embedding_dim=100, fourier_p=1.0, scale=1.0):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_params = model_params
        self.y_mean = y_mean
        self.y_std = y_std
        self.scale = scale
        self.inputs_ft = build_inputs_ft(T=200, embedding_dim=embedding_dim, p=fourier_p).to(self.device)
        self.hyper = HypyerNet(model_params).to(self.device)
        self.mixer = Mixer(types.SimpleNamespace(**model_params['mixer_args'])).to(self.device)
        self._load_weights(weights_folder)
        self.hyper.eval()
        self.mixer.eval()
        for p in self.hyper.parameters():
            p.requires_grad = False
        for p in self.mixer.parameters():
            p.requires_grad = False

    def _load_weights(self, folder):
        self.hyper.load_state_dict(torch.load(os.path.join(folder, 'hyper_best.pth'), map_location=self.device))
        self.mixer.load_state_dict(torch.load(os.path.join(folder, 'mixer_best.pth'), map_location=self.device))

    def make_inputs(self, x_batch):
        """
        x_batch: (B, 21)
        returns: (B, 200, 221)
        """
        x_rep = x_batch[:, None, :].repeat(1, 200, 1)
        ft_rep = self.inputs_ft.repeat(x_batch.shape[0], 1, 1)
        return torch.cat([ft_rep, x_rep], dim=-1)

    def forward_from_latent(self, x_batch, s_latent):
        """
        x_batch:  (B, 21)
        s_latent: (1, s_dim)
        returns:  (B, 200, 12)
        """
        inputs = self.make_inputs(x_batch)
        codes = self.mixer(self.scale * s_latent)
        if codes.ndim == 2:
            codes = codes.unsqueeze(1)
        output, _ = self.hyper(codes, inputs)
        if output.ndim == 4:
            output = output[0]
        loss_mean = output[..., 0:1]
        loss_cum_mean = torch.cumsum(loss_mean, dim=1)
        pred = torch.cat([output, loss_cum_mean], dim=-1)
        return pred

    def calibration_loss(self, x_batch, y_batch, s_latent, lambda_twiss=0.1, lambda_latent=0.0001, use_all_steps=False):
        pred = self.forward_from_latent(x_batch, s_latent)
        if use_all_steps:
            pred_for_fit = pred
            y_for_fit = y_batch
        else:
            pred_for_fit = pred[:, :, 0:1]
            y_for_fit = y_batch[:, :, 0:1]
        fit_loss = torch.mean((pred_for_fit - y_for_fit) ** 2)
        twiss_loss = twiss_rms_mismatch_np(pred[:, :, :], y_batch[:, :, :], self.y_mean, self.y_std).mean()
        latent_reg = (s_latent ** 2).mean()
        total = fit_loss + lambda_twiss * twiss_loss + lambda_latent * latent_reg
        return (total, {'fit_loss': fit_loss.detach().item(), 'twiss_loss': twiss_loss.detach().item(), 'latent_reg': latent_reg.detach().item()})

    def calibrate(self, x_val, y_val, n_restarts=8, adam_steps=300, lbfgs_steps=100, adam_lr=0.01, lambda_twiss=1.0, lambda_latent=0.0001, use_all_steps=False, batch_size=None):
        x_val = torch.as_tensor(x_val, dtype=torch.float32, device=self.device)
        y_val = torch.as_tensor(y_val, dtype=torch.float32, device=self.device)
        if batch_size is None:
            batch_size = len(x_val)
        best = {'loss': float('inf'), 'latent': None, 'stats': None}
        for restart in range(n_restarts):
            s_dim = self.model_params['mixer_args']['s']
            s_latent = torch.nn.Parameter(0.1 * torch.randn(1, s_dim, device=self.device))
            adam = torch.optim.Adam([s_latent], lr=adam_lr)
            for step in range(adam_steps):
                total_loss_accum = 0.0
                perm = torch.randperm(len(x_val), device=self.device)
                for i in range(0, len(x_val), batch_size):
                    idx = perm[i:i + batch_size]
                    xb = x_val[idx]
                    yb = y_val[idx]
                    adam.zero_grad()
                    loss, _ = self.calibration_loss(xb, yb, s_latent, lambda_twiss=lambda_twiss, lambda_latent=lambda_latent, use_all_steps=use_all_steps)
                    loss.backward()
                    adam.step()
                    total_loss_accum += loss.item() * len(idx)
            lbfgs = torch.optim.LBFGS([s_latent], lr=0.5, max_iter=lbfgs_steps, line_search_fn='strong_wolfe')

            def closure():
                lbfgs.zero_grad()
                loss, _ = self.calibration_loss(x_val, y_val, s_latent, lambda_twiss=lambda_twiss, lambda_latent=lambda_latent, use_all_steps=use_all_steps)
                loss.backward()
                return loss
            lbfgs.step(closure)
            with torch.no_grad():
                final_loss, stats = self.calibration_loss(x_val, y_val, s_latent, lambda_twiss=lambda_twiss, lambda_latent=lambda_latent, use_all_steps=use_all_steps)
            print(f"[restart {restart + 1}/{n_restarts}] total={final_loss.item():.6f} fit={stats['fit_loss']:.6f} twiss={stats['twiss_loss']:.6f}")
            if final_loss.item() < best['loss']:
                best['loss'] = final_loss.item()
                best['latent'] = s_latent.detach().clone()
                best['stats'] = stats
        return best

    @torch.no_grad()
    def export_generated_weights(self, s_latent, save_path):
        codes = self.mixer(self.scale * s_latent)
        if codes.ndim == 2:
            codes = codes.unsqueeze(1)
        bundle = {'codes': codes.cpu(), 'latent': s_latent.cpu()}
        torch.save(bundle, save_path)

    @torch.no_grad()
    def evaluate_latent(self, x_val, y_val, s_latent, lambda_twiss=1.0, lambda_latent=0.0001, use_all_steps=False, batch_size=256):
        x_val = torch.as_tensor(x_val, dtype=torch.float32, device=self.device)
        y_val = torch.as_tensor(y_val, dtype=torch.float32, device=self.device)
        total_fit = 0.0
        total_twiss = 0.0
        total_reg = 0.0
        total_total = 0.0
        n_total = 0
        for i in range(0, len(x_val), batch_size):
            xb = x_val[i:i + batch_size]
            yb = y_val[i:i + batch_size]
            total, stats = self.calibration_loss(xb, yb, s_latent, lambda_twiss=lambda_twiss, lambda_latent=lambda_latent, use_all_steps=use_all_steps)
            bsz = len(xb)
            total_fit += stats['fit_loss'] * bsz
            total_twiss += stats['twiss_loss'] * bsz
            total_reg += stats['latent_reg'] * bsz
            total_total += total.item() * bsz
            n_total += bsz
        return {'total_loss': total_total / n_total, 'fit_loss': total_fit / n_total, 'twiss_loss': total_twiss / n_total, 'latent_reg': total_reg / n_total}

    def initial_latent(self, mode='zero'):
        s_dim = self.model_params['mixer_args']['s']
        if mode == 'zero':
            return torch.zeros(1, s_dim, device=self.device)
        elif mode == 'random':
            return 0.1 * torch.randn(1, s_dim, device=self.device)
        else:
            raise ValueError("mode must be 'zero' or 'random'")

def print_comparison(before, after, name_before='Before calibration', name_after='After calibration'):
    print('\n' + '=' * 72)
    print(f'{name_before} vs {name_after}')
    print('=' * 72)
    for key in ['fit_loss', 'twiss_loss', 'total_loss']:
        b = before[key]
        a = after[key]
        abs_imp = b - a
        pct_imp = 100.0 * abs_imp / (abs(b) + 1e-12)
        print(f'{key:12s} | before = {b:.6f} | after = {a:.6f} | improvement = {abs_imp:.6f} ({pct_imp:.2f}%)')

def make_nested_calibration_splits(n_sample, n_test=128, calib_sizes=(64, 128, 256, 512, 1024), seed=42):
    """
    Create one fixed test index set and nested calibration index sets.

    Full index permutation:
        test_idx = first n_test samples
        calib_pool_idx = remaining samples

    Calibration sets are nested:
        calib-128  = first 128 from calib_pool_idx
        calib-256  = first 256 from calib_pool_idx
        calib-512  = first 512 from calib_pool_idx
        calib-1024 = first 1024 from calib_pool_idx
    """
    max_calib = max(calib_sizes)
    assert n_test + max_calib <= n_sample, f'Need at least {n_test + max_calib} samples, but only have {n_sample}.'
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_sample)
    test_idx = perm[:n_test]
    calib_pool_idx = perm[n_test:]
    calib_indices = {}
    for n_calib in calib_sizes:
        calib_indices[n_calib] = calib_pool_idx[:n_calib]
    return (test_idx, calib_indices, calib_pool_idx, perm)

@torch.no_grad()
def find_odd_samples(calibrator, x, y, s_latent, global_idx=None, top_k=20, use_all_steps=True, eps=0.0001):
    """
    Finds odd samples by per-sample fit loss and Twiss mismatch.

    x, y:
        Dataset arrays after normalisation and transpose.
        y shape should be (N, 200, 12).

    global_idx:
        Optional original indices in the full sim_good dataset.
        If provided, the printed indices will map back to the original data.
    """
    x_t = torch.as_tensor(x, dtype=torch.float32, device=calibrator.device)
    y_t = torch.as_tensor(y, dtype=torch.float32, device=calibrator.device)
    pred = calibrator.forward_from_latent(x_t, s_latent)
    if use_all_steps:
        fit_per_sample = ((pred - y_t) ** 2).mean(dim=(1, 2))
    else:
        fit_per_sample = ((pred[:, -1, :] - y_t[:, -1, :]) ** 2).mean(dim=1)
    pred_phys = unnormalize_y(pred, calibrator.y_mean, calibrator.y_std)
    y_phys = unnormalize_y(y_t, calibrator.y_mean, calibrator.y_std)
    ax_p, bx_p, gx_p = get_twiss_x(pred_phys, eps)
    ay_p, by_p, gy_p = get_twiss_y(pred_phys, eps)
    ax_t, bx_t, gx_t = get_twiss_x(y_phys, eps)
    ay_t, by_t, gy_t = get_twiss_y(y_phys, eps)
    mx_rms = 0.5 * (bx_t * gx_p - 2.0 * ax_t * ax_p + gx_t * bx_p)
    my_rms = 0.5 * (by_t * gy_p - 2.0 * ay_t * ay_p + gy_t * by_p)
    loss_x = torch.clamp(mx_rms - 1.0, min=0.0)
    loss_y = torch.clamp(my_rms - 1.0, min=0.0)
    twiss_step_loss = 0.5 * (loss_x + loss_y)
    if twiss_step_loss.ndim == 2:
        twiss_per_sample = twiss_step_loss.mean(dim=1)
        twiss_max_step = twiss_step_loss.max(dim=1).values
    else:
        twiss_per_sample = twiss_step_loss
        twiss_max_step = twiss_step_loss
    fit_np = fit_per_sample.detach().cpu().numpy()
    twiss_np = twiss_per_sample.detach().cpu().numpy()
    twiss_max_np = twiss_max_step.detach().cpu().numpy()
    if global_idx is None:
        global_idx = np.arange(len(x))
    else:
        global_idx = np.asarray(global_idx)
    print('\n' + '=' * 90)
    print('Per-sample loss summary')
    print('=' * 90)
    for name, arr in [('fit_loss', fit_np), ('twiss_loss', twiss_np), ('twiss_max_step', twiss_max_np)]:
        print(f'\n{name}')
        print(f'mean   = {arr.mean():.6f}')
        print(f'median = {np.median(arr):.6f}')
        print(f'std    = {arr.std():.6f}')
        print(f'min    = {arr.min():.6f}')
        print(f'max    = {arr.max():.6f}')
        print(f'p90    = {np.percentile(arr, 90):.6f}')
        print(f'p95    = {np.percentile(arr, 95):.6f}')
        print(f'p99    = {np.percentile(arr, 99):.6f}')
    worst_twiss = np.argsort(twiss_np)[-top_k:][::-1]
    worst_fit = np.argsort(fit_np)[-top_k:][::-1]
    print('\n' + '=' * 90)
    print(f'Top {top_k} samples by Twiss loss')
    print('=' * 90)
    for rank, local_i in enumerate(worst_twiss, 1):
        print(f'{rank:2d} | local_idx={local_i:5d} | global_idx={global_idx[local_i]:5d} | twiss={twiss_np[local_i]:12.6f} | twiss_max_step={twiss_max_np[local_i]:12.6f} | fit={fit_np[local_i]:12.6f}')
    print('\n' + '=' * 90)
    print(f'Top {top_k} samples by fit loss')
    print('=' * 90)
    for rank, local_i in enumerate(worst_fit, 1):
        print(f'{rank:2d} | local_idx={local_i:5d} | global_idx={global_idx[local_i]:5d} | fit={fit_np[local_i]:12.6f} | twiss={twiss_np[local_i]:12.6f} | twiss_max_step={twiss_max_np[local_i]:12.6f}')
    return {'fit': fit_np, 'twiss': twiss_np, 'twiss_max_step': twiss_max_np, 'global_idx': global_idx, 'worst_twiss_local': worst_twiss, 'worst_twiss_global': global_idx[worst_twiss], 'worst_fit_local': worst_fit, 'worst_fit_global': global_idx[worst_fit]}
if __name__ == '__main__':
    set_seed(42)
    data_path = 'cal_data'
    x_target, y_target, x_mean, x_std, y_mean, y_std = load_datasets(data_path)
    use_all_steps = False
    print(f'Total available target validation/good data: {len(x_target)}')
    calibration_sizes = [64, 128, 256, 512, 1024]
    model_params = {'in_args': {'z': 128, 'hidden_dim': 512, 'bias': False, 'in_feature': 21 + 100 * 2, 'out_feature': 128}, 'share_args': {'z': 128, 'hidden_dim': 512, 'bias': False, 'in_feature': 128, 'out_feature': 256}, 'b0_args': {'z': 128, 'hidden_dim': 512, 'bias': False, 'in_feature': 256, 'out_feature': 1}, 'b1_args': {'z': 128, 'hidden_dim': 512, 'bias': False, 'in_feature': 128, 'out_feature': 1}, 'd0_args': {'z': 128, 'hidden_dim': 512, 'bias': False, 'in_feature': 256, 'out_feature': 128}, 'd1_args': {'z': 128, 'hidden_dim': 512, 'bias': False, 'in_feature': 128, 'out_feature': 11}, 'd1q_args': {'z': 128, 'hidden_dim': 512, 'bias': False, 'in_feature': 128, 'out_feature': 11}, 'mixer_args': {'s': 256, 'z': 128, 'hidden_dim': 512, 'bias': False, 'n_layers': 5}, 'diser_args': {'z': 128, 'hidden_dim': 512}}
    calibrator = HyperCalibrator(model_params=model_params, weights_folder='best_physics_model', y_mean=y_mean, y_std=y_std, embedding_dim=100, fourier_p=1.0, scale=1000.0)
    lambda_twiss = 1.0
    lambda_latent = 0.001
    all_results = {}
    s_init = calibrator.initial_latent(mode='random')
    odd_info_all = find_odd_samples(calibrator=calibrator, x=x_target, y=y_target, s_latent=s_init, global_idx=np.arange(len(x_target)), top_k=30, use_all_steps=use_all_steps)
    bad_global_idx = np.array([927, 419, 970, 826, 84])
    keep_mask = np.ones(len(x_target), dtype=bool)
    keep_mask[bad_global_idx] = False
    x_target_clean = x_target[keep_mask]
    y_target_clean = y_target[keep_mask]
    print('Original size:', len(x_target))
    print('Clean size:', len(x_target_clean))
    print('Removed:', len(x_target) - len(x_target_clean))
    n_sample = len(x_target_clean)
    test_idx, calib_indices, calib_pool_idx, full_perm = make_nested_calibration_splits(n_sample=n_sample, n_test=128, calib_sizes=(64, 128, 256, 512, 1024), seed=42)
    x_test = x_target_clean[test_idx]
    y_test = y_target_clean[test_idx]
    print(f'Fixed test samples:      {len(x_test)}')
    before_test_stats = calibrator.evaluate_latent(x_val=x_test, y_val=y_test, s_latent=s_init, lambda_twiss=lambda_twiss, lambda_latent=lambda_latent, use_all_steps=use_all_steps, batch_size=256)
    print('\n' + '=' * 80)
    print('Before calibration on fixed 128-sample test set')
    print('=' * 80)
    print(before_test_stats)
    for n_calib in calibration_sizes:
        print('\n' + '=' * 80)
        print(f'Calibration experiment with n_calib = {n_calib}')
        print('=' * 80)
        idx = calib_indices[n_calib]
        x_calib = x_target_clean[idx]
        y_calib = y_target_clean[idx]
        print(f'\nCalibration size: {n_calib}')
        print(f'Test size:        {len(x_test)}')
        print(f'Calib size:       {len(x_calib)}')
        before_calib_stats = calibrator.evaluate_latent(x_val=x_calib, y_val=y_calib, s_latent=s_init, lambda_twiss=lambda_twiss, lambda_latent=lambda_latent, use_all_steps=use_all_steps, batch_size=256)
        calib_start_time = time.perf_counter()
        result = calibrator.calibrate(x_val=x_calib, y_val=y_calib, n_restarts=8, adam_steps=300, lbfgs_steps=200, adam_lr=0.01, lambda_twiss=lambda_twiss, lambda_latent=lambda_latent, use_all_steps=use_all_steps, batch_size=min(256, len(x_calib)))
        calib_end_time = time.perf_counter()
        calib_time_sec = calib_end_time - calib_start_time
        print(f'\nCalibration time for n_calib={n_calib}: {calib_time_sec:.2f} seconds')
        print(f'Calibration time for n_calib={n_calib}: {calib_time_sec / 60:.2f} minutes')
        print('\nBest calibration result')
        print('total loss:', result['loss'])
        print('stats:', result['stats'])
        after_calib_stats = calibrator.evaluate_latent(x_val=x_calib, y_val=y_calib, s_latent=result['latent'], lambda_twiss=lambda_twiss, lambda_latent=lambda_latent, use_all_steps=use_all_steps, batch_size=256)
        after_test_stats = calibrator.evaluate_latent(x_val=x_test, y_val=y_test, s_latent=result['latent'], lambda_twiss=lambda_twiss, lambda_latent=lambda_latent, use_all_steps=use_all_steps, batch_size=256)
        print('\nCalibration set improvement')
        print_comparison(before_calib_stats, after_calib_stats, name_before=f'Before on calib-{n_calib}', name_after=f'After on calib-{n_calib}')
        print('\nFixed test set improvement')
        print_comparison(before_test_stats, after_test_stats, name_before='Before on fixed test-128', name_after=f'After calib-{n_calib}')
        all_results[n_calib] = {'before_calib': before_calib_stats, 'after_calib': after_calib_stats, 'before_test': before_test_stats, 'after_test': after_test_stats, 'best_loss': result['loss'], 'best_stats': result['stats'], 'calib_indices': idx, 'calibration_time_sec': calib_time_sec}
        torch.save(result['latent'].cpu(), f'calibrated_latent_n{n_calib}.pt')
        calibrator.export_generated_weights(result['latent'], f'calibrated_model_bundle_n{n_calib}.pt')
    torch.save({'all_results': all_results, 'test_idx': test_idx, 'pool_idx': idx, 'calibration_sizes': calibration_sizes, 'lambda_twiss': lambda_twiss, 'lambda_latent': lambda_latent}, 'calibration_size_study_results.pt')
    print('\n' + '=' * 80)
    print('Summary: fixed test performance after calibration')
    print('=' * 80)
    for n_calib in calibration_sizes:
        stats = all_results[n_calib]['after_test']
        t_sec = all_results[n_calib]['calibration_time_sec']
        print(f"n_calib={n_calib:4d} | test_total={stats['total_loss']:.6f} | test_fit={stats['fit_loss']:.6f} | test_twiss={stats['twiss_loss']:.6f} | latent_reg={stats['latent_reg']:.6f} | time={t_sec:.2f}s")
