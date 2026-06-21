#!/usr/bin/env python3

"""
Generate shifted surrogate systems from a calibrated latent vector.

What this script does
1. Loads the pretrained hypernetwork + mixer.
2. Loads the calibrated latent z_ref (this defines S1).
3. Finds a physics-guided direction in latent space by maximising
   final-step Twiss mismatch sensitivity on a probe dataset.
4. Traverses the latent space along that direction to generate
   a sequence of related systems S1, S2, ..., SN.
5. Saves:
   - all latent vectors
   - per-system metrics relative to S1
   - fixed generated model bundles for downstream BO / RL studies

Notes
- "Physics-guided latent traversal" is a better name than just
  "physics-guided generation".
- This script uses the calibrated latent as S1.
- The last system in the sequence can be treated as S2 if you want
  a single source-target transfer study.
"""
import os
import json
import math
import types
import random
from dataclasses import dataclass, asdict
import numpy as np
import torch
import h5py
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
    x_train, y_train = load_h5(f'{save_dir}/sim_good_train.h5')
    x_val, y_val = load_h5(f'{save_dir}/sim_good_val.h5')
    x_mean, x_std = x_get_mean_std(x_train)
    x_train = (x_train - x_mean) / x_std
    x_val = (x_val - x_mean) / x_std
    y_mean, y_std = y_get_mean_std(y_train)
    y_train[:, 1:-1, :] = (y_train[:, 1:-1, :] - y_mean[:, 1:-1, :]) / y_std[:, 1:-1, :]
    y_val[:, 1:-1, :] = (y_val[:, 1:-1, :] - y_mean[:, 1:-1, :]) / y_std[:, 1:-1, :]
    y_train = np.transpose(y_train, (0, 2, 1))
    y_val = np.transpose(y_val, (0, 2, 1))
    return (x_train, y_train, x_val, y_val, x_mean, x_std, y_mean, y_std)

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
    sigma_x = torch.clamp(outputs[..., 3], min=eps)
    emit_x = torch.clamp(outputs[..., 5], min=eps)
    cov_xxp = outputs[..., 9]
    beta_x = torch.clamp(sigma_x ** 2 / emit_x, min=eps)
    alpha_x = -cov_xxp / torch.clamp(sigma_x ** 2, min=eps)
    gamma_x = (1.0 + alpha_x ** 2) / beta_x
    return (alpha_x, beta_x, gamma_x)

def get_twiss_y(outputs, eps=1e-10):
    sigma_y = torch.clamp(outputs[..., 4], min=eps)
    emit_y = torch.clamp(outputs[..., 6], min=eps)
    cov_yyp = outputs[..., 10]
    beta_y = torch.clamp(sigma_y ** 2 / emit_y, min=eps)
    alpha_y = -cov_yyp / torch.clamp(sigma_y ** 2, min=eps)
    gamma_y = (1.0 + alpha_y ** 2) / beta_y
    return (alpha_y, beta_y, gamma_y)

def twiss_rms_mismatch_torch(pred, target, y_mean, y_std, eps=0.0001):
    """
    pred, target: (..., 12) in normalised space
    Returns elementwise mismatch loss.
    """
    pred_phys = unnormalize_y(pred, y_mean, y_std)
    target_phys = unnormalize_y(target, y_mean, y_std)
    ax_p, bx_p, gx_p = get_twiss_x(pred_phys, eps)
    ay_p, by_p, gy_p = get_twiss_y(pred_phys, eps)
    ax_t, bx_t, gx_t = get_twiss_x(target_phys, eps)
    ay_t, by_t, gy_t = get_twiss_y(target_phys, eps)
    mx_rms = 0.5 * (bx_t * gx_p - 2.0 * ax_t * ax_p + gx_t * bx_p)
    my_rms = 0.5 * (by_t * gy_p - 2.0 * ay_t * ay_p + gy_t * by_p)
    loss_x = torch.clamp(mx_rms - 1.0, min=0.0)
    loss_y = torch.clamp(my_rms - 1.0, min=0.0)
    return 0.5 * (loss_x + loss_y)

def fit_mse(pred, target):
    return torch.mean((pred - target) ** 2)

def fit_nrmse(pred, target, eps=1e-08):
    return torch.sqrt(torch.mean((pred - target) ** 2) / (torch.mean(target ** 2) + eps))

class HyperSystemFactory:

    def __init__(self, model_params, weights_folder, y_mean, y_std, device=None, embedding_dim=100, fourier_p=1.0, scale=1000.0):
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

    def codes_from_latent(self, s_latent):
        """
        s_latent: (1, s_dim)
        returns codes in the shape expected by the hypernetwork.
        """
        codes = self.mixer(self.scale * s_latent)
        if codes.ndim == 2:
            codes = codes.unsqueeze(1)
        return codes

    def predict_from_latent(self, x_batch, s_latent):
        """
        x_batch: (B, 21)
        s_latent: (1, s_dim)
        returns: (B, 200, 12)
        """
        inputs = self.make_inputs(x_batch)
        codes = self.codes_from_latent(s_latent)
        output, _ = self.hyper(codes, inputs)
        if output.ndim == 4:
            output = output[0]
        loss_mean = output[..., 0:1]
        loss_cum_mean = torch.cumsum(loss_mean, dim=1)
        pred = torch.cat([output, loss_cum_mean], dim=-1)
        return pred

    @torch.no_grad()
    def evaluate_system(self, x_data, y_data, s_latent, batch_size=256):
        x_data = torch.as_tensor(x_data, dtype=torch.float32, device=self.device)
        y_data = torch.as_tensor(y_data, dtype=torch.float32, device=self.device)
        fit_all = 0.0
        fit_final = 0.0
        nrmse_final = 0.0
        twiss_final = 0.0
        n_total = 0
        for i in range(0, len(x_data), batch_size):
            xb = x_data[i:i + batch_size]
            yb = y_data[i:i + batch_size]
            pred = self.predict_from_latent(xb, s_latent)
            bsz = len(xb)
            fit_all += fit_mse(pred, yb).item() * bsz
            fit_final += fit_mse(pred[:, :, :], yb[:, :, :]).item() * bsz
            nrmse_final += fit_nrmse(pred[:, :, :], yb[:, :, :]).item() * bsz
            twiss_final += twiss_rms_mismatch_torch(pred[:, :, :], yb[:, :, :], self.y_mean, self.y_std).mean().item() * bsz
            n_total += bsz
        return {'fit_mse_all': fit_all / n_total, 'fit_mse_final': fit_final / n_total, 'fit_nrmse_final': nrmse_final / n_total, 'twiss_final': twiss_final / n_total}

    @torch.no_grad()
    def export_generated_weights(self, s_latent, save_path):
        codes = self.codes_from_latent(s_latent)
        w0, b0, w1, b1, wd0, bd0, wd1, bd1, wd1q, bd1q = self.hyper.get_weights(codes)
        bundle = {'w0': w0.detach().cpu(), 'b0': b0.detach().cpu(), 'w1': w1.detach().cpu(), 'b1': b1.detach().cpu(), 'wd0': wd0.detach().cpu(), 'bd0': bd0.detach().cpu(), 'wd1': wd1.detach().cpu(), 'bd1': bd1.detach().cpu(), 'wd1q': wd1q.detach().cpu(), 'bd1q': bd1q.detach().cpu(), 'codes': codes.detach().cpu(), 'latent': s_latent.detach().cpu()}
        torch.save(bundle, save_path)

@torch.no_grad()
def random_unit_vectors_like(z, n_samples):
    vecs = torch.randn(n_samples, *z.shape[1:], device=z.device)
    vecs = vecs / (torch.norm(vecs, dim=-1, keepdim=True) + 1e-12)
    return vecs

def find_twiss_sensitive_direction(factory, z_ref, x_probe, y_probe, n_candidates=512, eps=0.05, batch_size=256):
    """
    Finds a unit latent direction that maximally changes the final-step
    Twiss mismatch relative to the reference system.
    """
    device = factory.device
    x_probe = torch.as_tensor(x_probe, dtype=torch.float32, device=device)
    y_probe = torch.as_tensor(y_probe, dtype=torch.float32, device=device)
    vecs = random_unit_vectors_like(z_ref, n_candidates)
    best_score = -float('inf')
    best_v = None
    base_metrics = factory.evaluate_system(x_probe, y_probe, z_ref, batch_size=batch_size)
    base_twiss = base_metrics['twiss_final']
    print(f'Reference system Twiss loss on probe set: {base_twiss:.6f}')
    for i in range(n_candidates):
        v = vecs[i:i + 1]
        z_try = z_ref + eps * v
        metrics = factory.evaluate_system(x_probe, y_probe, z_try, batch_size=batch_size)
        delta_twiss = metrics['twiss_final'] - base_twiss
        score = abs(delta_twiss)
        if score > best_score:
            best_score = score
            best_v = v.clone()
    print(f'Best local Twiss sensitivity found: |ΔTwiss| = {best_score:.6f} for eps={eps:.4f}')
    return best_v

def build_latent_sequence(z_ref, direction, n_systems=10, max_radius=2.0, include_ref=True):
    """
    Generates a monotonic latent traversal:
      S1 = z_ref
      S2 = z_ref + alpha_1 * v
      ...
      SN = z_ref + alpha_{N-1} * v
    """
    if include_ref:
        alphas = torch.linspace(0.0, max_radius, steps=n_systems, device=z_ref.device)
    else:
        alphas = torch.linspace(max_radius / n_systems, max_radius, steps=n_systems, device=z_ref.device)
    latents = []
    for a in alphas:
        latents.append(z_ref + a * direction)
    return (alphas, latents)

def latent_distance(z_a, z_b):
    return torch.norm(z_a - z_b).item()

@torch.no_grad()
def choose_max_radius_by_target_twiss(factory, z_ref, direction, x_probe, y_probe, target_twiss, radius_lo=0.0, radius_hi=5.0, n_binary_steps=25, batch_size=256):
    """
    Chooses a radius so that the last system has approximately the target
    final-step Twiss loss on the probe set.
    """
    x_probe = torch.as_tensor(x_probe, dtype=torch.float32, device=factory.device)
    y_probe = torch.as_tensor(y_probe, dtype=torch.float32, device=factory.device)
    best_radius = radius_hi
    best_gap = float('inf')
    for _ in range(n_binary_steps):
        r = 0.5 * (radius_lo + radius_hi)
        z_try = z_ref + r * direction
        metrics = factory.evaluate_system(x_probe, y_probe, z_try, batch_size=batch_size)
        twiss_val = metrics['twiss_final']
        gap = abs(twiss_val - target_twiss)
        if gap < best_gap:
            best_gap = gap
            best_radius = r
        if twiss_val < target_twiss:
            radius_lo = r
        else:
            radius_hi = r
    return best_radius

@dataclass
class SystemRecord:
    system_name: str
    index: int
    alpha: float
    latent_distance_from_S1: float
    fit_mse_all: float
    fit_mse_final: float
    fit_nrmse_final: float
    twiss_final: float

def save_json(obj, path):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)

def save_records_csv(records, path):
    import csv
    if len(records) == 0:
        return
    fieldnames = list(records[0].keys())
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
if __name__ == '__main__':
    set_seed(42)
    data_path = 'sim_data'
    weights_folder = 'check_points'
    calibrated_latent_path = 'calibrated_latent.pt'
    out_dir = 'shifted_systems_twiss_guided'
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'bundles'), exist_ok=True)
    n_systems = 10
    direction_candidates = 512
    direction_eps = 0.05
    batch_size = 256
    use_target_twiss = False
    target_twiss_for_last = 1.5
    fixed_max_radius = 10.0
    model_params = {'in_args': {'z': 128, 'hidden_dim': 512, 'bias': False, 'in_feature': 21 + 100 * 2, 'out_feature': 128}, 'share_args': {'z': 128, 'hidden_dim': 512, 'bias': False, 'in_feature': 128, 'out_feature': 256}, 'b0_args': {'z': 128, 'hidden_dim': 512, 'bias': False, 'in_feature': 256, 'out_feature': 1}, 'b1_args': {'z': 128, 'hidden_dim': 512, 'bias': False, 'in_feature': 128, 'out_feature': 1}, 'd0_args': {'z': 128, 'hidden_dim': 512, 'bias': False, 'in_feature': 256, 'out_feature': 128}, 'd1_args': {'z': 128, 'hidden_dim': 512, 'bias': False, 'in_feature': 128, 'out_feature': 11}, 'd1q_args': {'z': 128, 'hidden_dim': 512, 'bias': False, 'in_feature': 128, 'out_feature': 11}, 'mixer_args': {'s': 256, 'z': 128, 'hidden_dim': 512, 'bias': False, 'n_layers': 5}, 'diser_args': {'z': 128, 'hidden_dim': 512}}
    x_train, y_train, x_val, y_val, x_mean, x_std, y_mean, y_std = load_datasets(data_path)
    factory = HyperSystemFactory(model_params=model_params, weights_folder=weights_folder, y_mean=y_mean, y_std=y_std, embedding_dim=100, fourier_p=1.0, scale=1000.0)
    z_ref = torch.load(calibrated_latent_path, map_location=factory.device)
    z_ref = torch.as_tensor(z_ref, dtype=torch.float32, device=factory.device)
    if z_ref.ndim == 1:
        z_ref = z_ref.unsqueeze(0)
    print('Loaded calibrated latent:', tuple(z_ref.shape))
    s1_metrics = factory.evaluate_system(x_val, y_val, z_ref, batch_size=batch_size)
    print('\nS1 metrics on val_good.h5')
    for k, v in s1_metrics.items():
        print(f'  {k}: {v:.6f}')
    direction = find_twiss_sensitive_direction(factory=factory, z_ref=z_ref, x_probe=x_val, y_probe=y_val, n_candidates=direction_candidates, eps=direction_eps, batch_size=batch_size)
    if use_target_twiss:
        max_radius = choose_max_radius_by_target_twiss(factory=factory, z_ref=z_ref, direction=direction, x_probe=x_val, y_probe=y_val, target_twiss=target_twiss_for_last, radius_lo=0.0, radius_hi=5.0, n_binary_steps=25, batch_size=batch_size)
        print(f'Chosen max_radius by target Twiss: {max_radius:.6f}')
    else:
        max_radius = fixed_max_radius
        print(f'Using fixed max_radius: {max_radius:.6f}')
    alphas, latents = build_latent_sequence(z_ref=z_ref, direction=direction, n_systems=n_systems, max_radius=max_radius, include_ref=True)
    summary_records = []
    latent_save = {}
    for idx, (alpha, z_i) in enumerate(zip(alphas, latents), start=1):
        system_name = f'S{idx}'
        metrics = factory.evaluate_system(x_val, y_val, z_i, batch_size=batch_size)
        dist = latent_distance(z_ref, z_i)
        rec = SystemRecord(system_name=system_name, index=idx, alpha=float(alpha.item()), latent_distance_from_S1=dist, fit_mse_all=metrics['fit_mse_all'], fit_mse_final=metrics['fit_mse_final'], fit_nrmse_final=metrics['fit_nrmse_final'], twiss_final=metrics['twiss_final'])
        summary_records.append(asdict(rec))
        latent_save[system_name] = z_i.detach().cpu()
        bundle_path = os.path.join(out_dir, 'bundles', f'{system_name}_bundle.pt')
        factory.export_generated_weights(z_i, bundle_path)
        print(f"{system_name:>3s} | alpha={float(alpha):.4f} | latent_dist={dist:.4f} | fit_final={metrics['fit_mse_final']:.6f} | twiss_final={metrics['twiss_final']:.6f}")
    torch.save(latent_save, os.path.join(out_dir, 'latents.pt'))
    metadata = {'description': 'Physics-guided latent traversal from calibrated S1 using Twiss-sensitive direction.', 'weights_folder': weights_folder, 'calibrated_latent_path': calibrated_latent_path, 'n_systems': n_systems, 'direction_candidates': direction_candidates, 'direction_eps': direction_eps, 'max_radius': max_radius, 'use_target_twiss': use_target_twiss, 'target_twiss_for_last': target_twiss_for_last if use_target_twiss else None, 'summary': summary_records}
    save_json(metadata, os.path.join(out_dir, 'metadata.json'))
    save_records_csv(summary_records, os.path.join(out_dir, 'system_summary.csv'))
    print('\nDone.')
    print(f"Saved latents to: {os.path.join(out_dir, 'latents.pt')}")
    print(f"Saved summary to: {os.path.join(out_dir, 'system_summary.csv')}")
    print(f"Saved metadata to: {os.path.join(out_dir, 'metadata.json')}")
    print(f"Saved bundles to: {os.path.join(out_dir, 'bundles')}")
    import pandas as pd
    import matplotlib.pyplot as plt
    df = pd.DataFrame(summary_records)
    plt.figure(figsize=(7, 4))
    plt.plot(df['index'], df['fit_mse_final'], marker='o', label='MSE')
    plt.xlabel('Model index')
    plt.ylabel('MSE')
    plt.title('MSE change along model index')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'mse_vs_index.png'), dpi=200)
    plt.close()
    plt.figure(figsize=(7, 4))
    plt.plot(df['index'], df['twiss_final'], marker='o', label='Twiss')
    plt.xlabel('Model index')
    plt.ylabel('Twiss')
    plt.title('Twiss change along model index')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'twiss_vs_index.png'), dpi=200)
    plt.close()
