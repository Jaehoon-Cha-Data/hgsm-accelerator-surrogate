#!/usr/bin/env python3

"""Train the mean-prediction HGSM/HyperGAN baseline.

This script is a cleaned, reviewer-facing version of the original
`M_runs_first_only_mean.py` training script. It keeps the same modelling
workflow and loss structure, but organises the code into small functions and a
trainer class so that the experiment can be inspected and rerun more easily.

Expected input files
--------------------
The data directory must contain the following HDF5 files:

    sim_good_train.h5
    sim_good_val.h5
    sim_good_test.h5
    sim_bad.h5

Each HDF5 file should contain `inputs` and `outputs` datasets. The bad-data file
should additionally contain `outputs_mask` and `outputs_info`.

Main outputs
------------
The script writes model checkpoints, diagnostic plots, and loss histories to:

    <output_root>/<model_name>_<ed_multiple>_<embedding_dim>_<beta>_<seed>/
"""
from __future__ import annotations
import argparse
import json
import pickle
import random
import types
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple
import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from models_first import Discriminator, HypyerNet, Mixer

@dataclass
class TrainingConfig:
    """Configuration for the mean HyperGAN training run."""
    path_dir: Path = Path.cwd()
    model_name: str = 'FF_Hyper_MAE_Loss'
    dataset: str = 'mebt'
    epochs: int = 1000
    n_models: int = 100
    batch_size: int = 64
    lr: float = 0.001
    lr_decay: float = 0.95
    seed: int = 0
    embedding_dim: int = 100
    beta: float = 50.0
    ed_multiple: int = 1
    include_bad: bool = True
    draw_every: int = 200
    output_root: Path = Path('results_new_bad_true_1000_epoch')
    input_noise_std: float = 0.005
    fourier_key: str = '1.0'

    @property
    def data_dir(self) -> Path:
        return self.path_dir / 'sim_data'

    @property
    def run_dir(self) -> Path:
        name = f'{self.model_name}_{self.ed_multiple}_{self.embedding_dim}_{self.beta}_{self.seed}'
        return self.path_dir / self.output_root / name

def parse_args() -> TrainingConfig:
    parser = argparse.ArgumentParser(description='Train the mean-prediction HGSM/HyperGAN baseline.')
    parser.add_argument('--path-dir', type=Path, default=Path.cwd())
    parser.add_argument('--model-name', type=str, default='FF_Hyper_MAE_Loss')
    parser.add_argument('--dataset', type=str, default='mebt')
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--n-models', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lr-decay', type=float, default=0.95)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--embedding-dim', type=int, default=100)
    parser.add_argument('--beta', type=float, default=50.0)
    parser.add_argument('--ed-multiple', type=int, default=1)
    parser.add_argument('--output-root', type=Path, default=Path('results_new_bad_true_1000_epoch'))
    parser.add_argument('--draw-every', type=int, default=200)
    parser.add_argument('--input-noise-std', type=float, default=0.005)
    parser.add_argument('--fourier-key', type=str, default='1.0')
    parser.add_argument('--exclude-bad', action='store_true', help='Train only with complete simulations and exclude bad/padded samples.')
    args = parser.parse_args()
    return TrainingConfig(path_dir=args.path_dir, model_name=args.model_name, dataset=args.dataset, epochs=args.epochs, n_models=args.n_models, batch_size=args.batch_size, lr=args.lr, lr_decay=args.lr_decay, seed=args.seed, embedding_dim=args.embedding_dim, beta=args.beta, ed_multiple=args.ed_multiple, include_bad=not args.exclude_bad, draw_every=args.draw_every, output_root=args.output_root, input_noise_std=args.input_noise_std, fourier_key=args.fourier_key)

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def enable_deterministic_mode(seed: int) -> None:
    set_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_h5(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, 'r') as h5:
        x = h5['inputs'][:].astype(np.float32)
        y = h5['outputs'][:].astype(np.float32)
    return (x, y)

def x_get_mean_std(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    std = np.where(std == 0.0, 1.0, std)
    return (mean, std)

def y_get_mean_std(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = np.mean(data, axis=(0, 2), keepdims=True)
    std = np.std(data, axis=(0, 2), keepdims=True)
    std = np.where(std == 0.0, 1.0, std)
    return (mean, std)

def normalise_outputs(y: np.ndarray, y_mean: np.ndarray, y_std: np.ndarray) -> np.ndarray:
    """Normalise descriptor channels while keeping loss channels in native scale."""
    y = y.copy()
    y[:, 1:-1, :] = (y[:, 1:-1, :] - y_mean[:, 1:-1, :]) / y_std[:, 1:-1, :]
    return y

def load_datasets(data_dir: Path) -> Dict[str, np.ndarray]:
    """Load and normalise train, validation, test, and bad simulation data."""
    x_train, y_train = load_h5(data_dir / 'sim_good_train.h5')
    x_val, y_val = load_h5(data_dir / 'sim_good_val.h5')
    x_test, y_test = load_h5(data_dir / 'sim_good_test.h5')
    with h5py.File(data_dir / 'sim_bad.h5', 'r') as h5:
        x_bad = h5['inputs'][:].astype(np.float32)
        y_bad = h5['outputs'][:].astype(np.float32)
        y_bad_info = h5['outputs_info'][:].astype(np.float32)
        y_bad_mask = h5['outputs_mask'][:].astype(np.float32)
    x_mean, x_std = x_get_mean_std(x_train)
    y_mean, y_std = y_get_mean_std(y_train)
    datasets = {'train_x': (x_train - x_mean) / x_std, 'val_x': (x_val - x_mean) / x_std, 'test_x': (x_test - x_mean) / x_std, 'bad_x': (x_bad - x_mean) / x_std, 'train_y': np.transpose(normalise_outputs(y_train, y_mean, y_std), (0, 2, 1)), 'val_y': np.transpose(normalise_outputs(y_val, y_mean, y_std), (0, 2, 1)), 'test_y': np.transpose(normalise_outputs(y_test, y_mean, y_std), (0, 2, 1)), 'bad_y': np.transpose(normalise_outputs(y_bad, y_mean, y_std), (0, 2, 1)), 'bad_mask': np.transpose(y_bad_mask, (0, 2, 1)), 'bad_info': y_bad_info, 'x_mean': x_mean, 'x_std': x_std, 'y_mean': y_mean, 'y_std': y_std}
    return datasets

def make_repeated_input(x: np.ndarray) -> torch.Tensor:
    """Convert static machine settings `(N, D)` to sequence inputs `(N, 200, D)`."""
    x_tensor = torch.from_numpy(np.expand_dims(x, axis=1))
    return x_tensor.repeat(1, 200, 1)

def build_dataloaders(data: Dict[str, np.ndarray], batch_size: int, include_bad: bool) -> Tuple[Dict[str, torch.Tensor], Dict[str, DataLoader]]:
    """Create tensors and PyTorch dataloaders used by the trainer."""
    tensors: Dict[str, torch.Tensor] = {'train_in': make_repeated_input(data['train_x']), 'val_in': make_repeated_input(data['val_x']), 'test_in': make_repeated_input(data['test_x']), 'bad_in': make_repeated_input(data['bad_x']), 'train_out': torch.from_numpy(data['train_y']), 'val_out': torch.from_numpy(data['val_y']), 'test_out': torch.from_numpy(data['test_y']), 'bad_out': torch.from_numpy(data['bad_y']), 'bad_mask': torch.from_numpy(data['bad_mask']), 'bad_info': torch.from_numpy(data['bad_info'])}
    if include_bad:
        train_inputs = torch.cat([tensors['train_in'], tensors['bad_in']], dim=0)
        train_outputs = torch.cat([tensors['train_out'], tensors['bad_out']], dim=0)
        train_mask = torch.cat([torch.ones_like(tensors['train_out']), tensors['bad_mask']], dim=0)
        train_dataset = TensorDataset(train_inputs, train_outputs, train_mask)
    else:
        train_dataset = TensorDataset(tensors['train_in'], tensors['train_out'])
    dataloaders = {'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True), 'val': DataLoader(TensorDataset(tensors['val_in'], tensors['val_out']), batch_size=batch_size, shuffle=False), 'test': DataLoader(TensorDataset(tensors['test_in'], tensors['test_out']), batch_size=batch_size, shuffle=False), 'bad': DataLoader(TensorDataset(tensors['bad_in'], tensors['bad_out'], tensors['bad_mask'], tensors['bad_info']), batch_size=batch_size, shuffle=False)}
    return (tensors, dataloaders)

def build_model_params(config: TrainingConfig) -> Dict[str, Dict[str, int | float | bool]]:
    input_dim = 21 + config.embedding_dim * 2
    return {'in_args': {'z': 128, 'hidden_dim': 512, 'bias': False, 'in_feature': input_dim, 'out_feature': 128}, 'share_args': {'z': 128, 'hidden_dim': 512, 'bias': False, 'in_feature': 128, 'out_feature': 256}, 'b0_args': {'z': 128, 'hidden_dim': 512, 'bias': False, 'in_feature': 256, 'out_feature': 1}, 'b1_args': {'z': 128, 'hidden_dim': 512, 'bias': False, 'in_feature': 128, 'out_feature': 1}, 'd0_args': {'z': 128, 'hidden_dim': 512, 'bias': False, 'in_feature': 256, 'out_feature': 128}, 'd1_args': {'z': 128, 'hidden_dim': 512, 'bias': False, 'in_feature': 128, 'out_feature': 11}, 'mixer_args': {'s': 256, 'z': 128, 'hidden_dim': 512, 'bias': False, 'n_layers': 4}, 'diser_args': {'z': 128, 'hidden_dim': 512}}

def make_fourier_features(embedding_dim: int, ed_multiple: int, key: str='1.0', n_steps: int=200) -> torch.Tensor:
    """Build deterministic Fourier features along the beamline coordinate."""
    location = np.float32(np.linspace(0.0, 1.0, n_steps, endpoint=False))
    bvals = np.float32(np.arange(1, embedding_dim * ed_multiple + 1))
    powers = [0.0, 0.5, 1.0, 1.5, 2.0]
    ab_dict = {str(p): (bvals ** (-np.float32(p)), bvals) for p in powers}
    if key not in ab_dict:
        raise ValueError(f'Unsupported Fourier key {key!r}; choose from {list(ab_dict)}')
    amplitudes, frequencies = ab_dict[key]
    encoded = np.concatenate([amplitudes * np.sin(2.0 * np.pi * location[..., None] * frequencies), amplitudes * np.cos(2.0 * np.pi * location[..., None] * frequencies)], axis=-1)
    encoded = encoded / np.linalg.norm(amplitudes)
    encoded = encoded[None, :, ::ed_multiple].astype(np.float32)
    return torch.from_numpy(encoded)

class HyperGANMeanTrainer:
    """Train and evaluate the mean-prediction HGSM/HyperGAN baseline."""

    def __init__(self, config: TrainingConfig, data: Dict[str, np.ndarray], tensors: Dict[str, torch.Tensor], dataloaders: Dict[str, DataLoader]) -> None:
        self.config = config
        self.data = data
        self.tensors = tensors
        self.dataloaders = dataloaders
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_params = build_model_params(config)
        self.inputs_ft = make_fourier_features(embedding_dim=config.embedding_dim, ed_multiple=config.ed_multiple, key=config.fourier_key).to(self.device)
        self.hyper = HypyerNet(self.model_params).to(self.device)
        self.mixer = Mixer(types.SimpleNamespace(**self.model_params['mixer_args'])).to(self.device)
        self.discriminator = Discriminator(types.SimpleNamespace(**self.model_params['diser_args'])).to(self.device)
        self.optimizer_generator = torch.optim.Adam(list(self.hyper.parameters()) + list(self.mixer.parameters()), lr=config.lr, betas=(0.9, 0.99))
        self.optimizer_discriminator = torch.optim.Adam(self.discriminator.parameters(), lr=config.lr * 10.0, betas=(0.5, 0.9))
        self.reconstruction_weights = torch.tensor([config.beta] + [1.0] * 11, dtype=torch.float32, device=self.device).view(1, 1, 1, -1)

    def _latent_samples(self) -> Tuple[torch.Tensor, torch.Tensor]:
        s = torch.randn(self.config.n_models, self.model_params['mixer_args']['s'], device=self.device)
        z = torch.randn(self.config.n_models, self.model_params['diser_args']['z'], device=self.device)
        return (s, z)

    def _make_inputs(self, data: torch.Tensor, add_noise: bool) -> torch.Tensor:
        if add_noise and self.config.input_noise_std > 0:
            data = data + self.config.input_noise_std * torch.randn_like(data)
        fourier = self.inputs_ft.repeat(data.shape[0], 1, 1)
        return torch.cat([fourier, data], dim=-1)

    def discriminator_loss(self, pred_fake: torch.Tensor, pred_real: torch.Tensor) -> torch.Tensor:
        target_real = torch.ones_like(pred_real, device=self.device)
        target_fake = torch.zeros_like(pred_fake, device=self.device)
        real_loss = nn.MSELoss()(pred_real, target_real)
        fake_loss = nn.MSELoss()(pred_fake, target_fake)
        return 0.5 * (real_loss + fake_loss)

    def train_epoch(self, epoch: int) -> float:
        self.hyper.train()
        self.mixer.train()
        self.discriminator.train()
        running_reconstruction_loss = 0.0
        train_loader = self.dataloaders['train']
        for samples in train_loader:
            data = samples[0].to(self.device)
            target = samples[1].to(self.device)
            mask = samples[2].to(self.device) if self.config.include_bad else None
            s, z = self._latent_samples()
            inputs = self._make_inputs(data, add_noise=True)
            codes = self.mixer(s)
            pred_codes_detached = self.discriminator(codes.detach())
            pred_z = self.discriminator(z)
            loss_d = self.discriminator_loss(pred_codes_detached, pred_z)
            self.optimizer_discriminator.zero_grad(set_to_none=True)
            loss_d.backward()
            self.optimizer_discriminator.step()
            pred_codes = self.discriminator(codes)
            output = self.hyper(codes, inputs)
            cumulative_loss = torch.cumsum(output[..., 0:1], dim=2)
            all_output = torch.cat([output, cumulative_loss], dim=-1)
            target_expanded = target.unsqueeze(0).repeat(self.config.n_models, 1, 1, 1)
            if mask is not None:
                mask_expanded = mask.unsqueeze(0).repeat(self.config.n_models, 1, 1, 1)
                all_output = all_output * mask_expanded
                target_expanded = target_expanded * mask_expanded
            target_codes = torch.ones_like(pred_codes, device=self.device)
            loss_g = nn.MSELoss()(pred_codes, target_codes)
            loss_r = nn.L1Loss()(all_output * self.reconstruction_weights, target_expanded * self.reconstruction_weights)
            loss = loss_g + loss_r
            self.optimizer_generator.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer_generator.step()
            running_reconstruction_loss += loss_r.detach().item()
        mean_loss = running_reconstruction_loss / len(train_loader)
        print(f'Train Epoch: {epoch}/{self.config.epochs} Loss: {mean_loss:.4f}')
        return mean_loss

    @torch.no_grad()
    def validate(self, epoch: int) -> float:
        self.hyper.eval()
        self.mixer.eval()
        running_loss = 0.0
        val_loader = self.dataloaders['val']
        for data, target in val_loader:
            data = data.to(self.device)
            target = target.to(self.device)
            inputs = self._make_inputs(data, add_noise=False)
            s, _ = self._latent_samples()
            codes = self.mixer(s)
            output = self.hyper(codes, inputs)
            target_expanded = target[..., 0:1].unsqueeze(0).repeat(self.config.n_models, 1, 1, 1)
            loss = nn.MSELoss()(output[..., 0:1], target_expanded)
            running_loss += loss.detach().item()
        mean_loss = running_loss / len(val_loader)
        print(f'Validation Epoch: {epoch}/{self.config.epochs} Loss: {mean_loss:.4f}')
        return mean_loss

    @torch.no_grad()
    def predict(self, data: torch.Tensor) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.hyper.eval()
        self.mixer.eval()
        data = data.to(self.device)
        inputs = self._make_inputs(data, add_noise=False)
        s, _ = self._latent_samples()
        codes = self.mixer(s)
        output = self.hyper(codes, inputs)
        cumulative_loss = torch.cumsum(output[..., 0:1], dim=2)
        all_output = torch.cat([output, cumulative_loss], dim=-1).cpu().numpy()
        mean_pred = np.mean(all_output, axis=0)
        lower = np.percentile(all_output, 5, axis=0)
        upper = np.percentile(all_output, 95, axis=0)
        return (mean_pred, lower, upper)

    def _plot_predictions(self, target: torch.Tensor, mean_pred: np.ndarray, lower: np.ndarray, upper: np.ndarray, save_path: Path, info: torch.Tensor | None=None) -> None:
        x_range = np.arange(mean_pred.shape[1])
        sample_idx = 0
        fig, axes = plt.subplots(6, 2, figsize=(12, 15))
        for output_idx, ax in enumerate(axes.flatten()):
            ax.plot(target[sample_idx, :, output_idx].cpu().numpy(), label='Target')
            ax.plot(mean_pred[sample_idx, :, output_idx], label='Prediction')
            ax.fill_between(x_range, lower[sample_idx, :, output_idx], upper[sample_idx, :, output_idx], label='5-95 percentile interval', alpha=0.2)
            if info is not None:
                stop_index = float(info[sample_idx, 0].item())
                final_loss = float(info[sample_idx, 1].item())
                ax.axvline(x=stop_index, linestyle='--', label='Final simulated step')
                ax.axvspan(stop_index, len(x_range), alpha=0.2, label='Padded region')
                if output_idx == 11:
                    ax.axhline(y=final_loss, linestyle='--', label='Final cumulative loss')
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=3)
        fig.tight_layout()
        fig.savefig(save_path, bbox_inches='tight', dpi=200)
        plt.close(fig)

    def draw_diagnostics(self, label: str | int) -> None:
        self.config.run_dir.mkdir(parents=True, exist_ok=True)
        batch_size = 10
        split_tensors = {'train': (self.tensors['train_in'], self.tensors['train_out'], None), 'val': (self.tensors['val_in'], self.tensors['val_out'], None), 'test': (self.tensors['test_in'], self.tensors['test_out'], None), 'bad': (self.tensors['bad_in'], self.tensors['bad_out'], self.tensors['bad_info'])}
        for split, (inputs, target, info) in split_tensors.items():
            inputs_batch = inputs[:batch_size]
            target_batch = target[:batch_size]
            info_batch = info[:batch_size] if info is not None else None
            mean_pred, lower, upper = self.predict(inputs_batch)
            self._plot_predictions(target=target_batch, mean_pred=mean_pred, lower=lower, upper=upper, save_path=self.config.run_dir / f'{split}_{label}.png', info=info_batch)

    def save_checkpoint(self, suffix: str) -> None:
        self.config.run_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.hyper.state_dict(), self.config.run_dir / f'hyper_{suffix}.pth')
        torch.save(self.mixer.state_dict(), self.config.run_dir / f'mixer_{suffix}.pth')
        torch.save(self.discriminator.state_dict(), self.config.run_dir / f'diser_{suffix}.pth')

    def save_config(self) -> None:
        self.config.run_dir.mkdir(parents=True, exist_ok=True)
        serialisable_config = asdict(self.config)
        serialisable_config['path_dir'] = str(serialisable_config['path_dir'])
        serialisable_config['output_root'] = str(serialisable_config['output_root'])
        with open(self.config.run_dir / 'training_config.json', 'w', encoding='utf-8') as f:
            json.dump(serialisable_config, f, indent=2)

    def run(self) -> Tuple[Iterable[float], Iterable[float]]:
        self.save_config()
        train_losses = []
        validation_losses = []
        best_validation_loss = float('inf')
        for epoch in range(self.config.epochs):
            train_loss = self.train_epoch(epoch)
            validation_loss = self.validate(epoch)
            train_losses.append(train_loss)
            validation_losses.append(validation_loss)
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                self.save_checkpoint('best')
                self.draw_diagnostics('best')
            if self.config.draw_every > 0 and epoch % self.config.draw_every == 0:
                self.draw_diagnostics(epoch)
        self.save_checkpoint('final')
        self.draw_diagnostics('final')
        return (train_losses, validation_losses)

def print_dataset_summary(data: Dict[str, np.ndarray]) -> None:
    for name in ['train', 'val', 'test', 'bad']:
        print(f"{name}_x shape: {data[f'{name}_x'].shape}")
        print(f"{name}_y shape: {data[f'{name}_y'].shape}")

def main() -> None:
    config = parse_args()
    enable_deterministic_mode(config.seed)
    config.run_dir.mkdir(parents=True, exist_ok=True)
    data = load_datasets(config.data_dir)
    print_dataset_summary(data)
    tensors, dataloaders = build_dataloaders(data=data, batch_size=config.batch_size, include_bad=config.include_bad)
    trainer = HyperGANMeanTrainer(config, data, tensors, dataloaders)
    train_losses, validation_losses = trainer.run()
    with open(config.run_dir / 'train_loss.pkl', 'wb') as f:
        pickle.dump(train_losses, f)
    with open(config.run_dir / 'validation_loss.pkl', 'wb') as f:
        pickle.dump(validation_losses, f)
if __name__ == '__main__':
    main()
