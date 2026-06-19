#!/usr/bin/env python3

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import matplotlib.pyplot as plt
import argparse
from collections import OrderedDict
import types
import pickle
import random
from models_second import Mixer, Discriminator, HypyerNet
import h5py
np.random.seed(1234)
torch.manual_seed(1234)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_dir', type=str, default=os.getcwd())
    parser.add_argument('--model_name', type=str, default='FF_Hyper_MAE_Loss')
    parser.add_argument('--dataset', type=str, default='mebt')
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--n_models', type=int, default=100, help='the number of models for training')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.0001, help='0.005, 0.001, 0.0005, 0.0001 can be options')
    parser.add_argument('--lr_decay', type=float, default=0.95)
    parser.add_argument('--rnd', type=float, default=0)
    parser.add_argument('--mode', type=str, default='train', help='[train or inference]')
    parser.add_argument('--embedding_dim', type=int, default=100, help='embedding dimension for the FFE')
    parser.add_argument('--beta', type=float, default=50.0, help='beta')
    parser.add_argument('--beta_idx', type=int, default=0, help='beta_idx')
    parser.add_argument('--ed_multiple', type=int, default=1, help='multiple of embedding dim')
    args = parser.parse_args()
    config = OrderedDict([('path_dir', args.path_dir), ('model_name', args.model_name), ('dataset', args.dataset), ('epochs', args.epochs), ('n_models', args.n_models), ('batch_size', args.batch_size), ('lr', args.lr), ('lr_decay', args.lr_decay), ('rnd', args.rnd), ('mode', args.mode), ('embedding_dim', args.embedding_dim), ('beta', args.beta), ('beta_idx', args.beta_idx), ('ed_multiple', args.ed_multiple)])
    return config

def main():
    config = parse_args()

    def set_seed(seed: int):
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    def reproducibility(seed: int):
        set_seed(seed)
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
        """
        train_x shape: (1617, 17) -> (2398, 21)
        train_y shape: (1617, 12, 200) -> (2398, 12, 200)
        val_x shape: (346, 17) -> (685, 21)
        val_y shape: (346, 12, 200) -> (685, 12, 200)
        test_x shape: (347, 17) -> (344, 21)
        test_y shape: (347, 12, 200) -> (344, 12, 200)
        """
        x_train, y_train = load_h5(f'{save_dir}/sim_good_train.h5')
        x_val, y_val = load_h5(f'{save_dir}/sim_good_val.h5')
        x_test, y_test = load_h5(f'{save_dir}/sim_good_test.h5')
        with h5py.File(f'{save_dir}/sim_bad.h5', 'r') as h5:
            x_bad = h5['inputs'][:].astype(np.float32)
            y_bad = h5['outputs'][:].astype(np.float32)
            y_bad_info = h5['outputs_info'][:].astype(np.float32)
            y_bad_mask = h5['outputs_mask'][:].astype(np.float32)
        x_mean, x_std = x_get_mean_std(x_train)
        x_train = (x_train - x_mean) / x_std
        x_val = (x_val - x_mean) / x_std
        x_test = (x_test - x_mean) / x_std
        x_bad = (x_bad - x_mean) / x_std
        y_mean, y_std = y_get_mean_std(y_train)
        y_train[:, 1:-1, :] = (y_train[:, 1:-1, :] - y_mean[:, 1:-1, :]) / y_std[:, 1:-1, :]
        y_val[:, 1:-1, :] = (y_val[:, 1:-1, :] - y_mean[:, 1:-1, :]) / y_std[:, 1:-1, :]
        y_test[:, 1:-1, :] = (y_test[:, 1:-1, :] - y_mean[:, 1:-1, :]) / y_std[:, 1:-1, :]
        y_bad[:, 1:-1, :] = (y_bad[:, 1:-1, :] - y_mean[:, 1:-1, :]) / y_std[:, 1:-1, :]
        return (x_train, np.transpose(y_train[:, :, :], (0, 2, 1)), x_val, np.transpose(y_val[:, :, :], (0, 2, 1)), x_test, np.transpose(y_test[:, :, :], (0, 2, 1)), x_bad, np.transpose(y_bad[:, :, :], (0, 2, 1)), np.transpose(y_bad_mask[:, :, :], (0, 2, 1)), y_bad_info)
    data_path = os.path.join(config['path_dir'], 'sim_data')
    train_x, train_y, val_x, val_y, test_x, test_y, bad_x, bad_y, bad_y_mask, bad_y_info = load_datasets(data_path)
    print('train_x shape:', train_x.shape)
    print('train_y shape:', train_y.shape)
    print('val_x shape:', val_x.shape)
    print('val_y shape:', val_y.shape)
    print('test_x shape:', test_x.shape)
    print('test_y shape:', test_y.shape)
    print('bad_x shape:', bad_x.shape)
    print('bad_y shape:', bad_y.shape)
    bad_in_var = np.expand_dims(bad_x, 1)
    bad_in_var = torch.from_numpy(bad_in_var)
    bad_in_var = bad_in_var.repeat(1, 200, 1)
    bad_out_var = bad_y
    bad_out_var = torch.from_numpy(bad_out_var)
    bad_mask_var = bad_y_mask
    bad_mask_var = torch.from_numpy(bad_mask_var)
    bad_info_var = bad_y_info
    bad_info_var = torch.from_numpy(bad_info_var)
    bad_dataset = TensorDataset(bad_in_var, bad_out_var, bad_mask_var, bad_info_var)
    bad_included = 'True'
    if bad_included == 'True':
        train_in_var = np.expand_dims(np.concatenate((train_x, bad_x), axis=0), 1)
        train_in_var = torch.from_numpy(train_in_var)
        train_in_var = train_in_var.repeat(1, 200, 1)
        train_out_var = np.concatenate((train_y, bad_y), axis=0)
        train_out_var = torch.from_numpy(train_out_var)
        train_mask_var = np.concatenate((np.ones_like(train_y), bad_y_mask), axis=0)
        train_mask_var = torch.from_numpy(train_mask_var)
        train_dataset = TensorDataset(train_in_var, train_out_var, train_mask_var)
    else:
        train_in_var = np.expand_dims(train_x, 1)
        train_in_var = torch.from_numpy(train_in_var)
        train_in_var = train_in_var.repeat(1, 200, 1)
        train_out_var = train_y
        train_out_var = torch.from_numpy(train_out_var)
        train_dataset = TensorDataset(train_in_var, train_out_var)
    val_in_var = np.expand_dims(val_x, 1)
    val_in_var = torch.from_numpy(val_in_var)
    val_in_var = val_in_var.repeat(1, 200, 1)
    val_out_var = val_y
    val_out_var = torch.from_numpy(val_out_var)
    val_dataset = TensorDataset(val_in_var, val_out_var)
    test_in_var = np.expand_dims(test_x, 1)
    test_in_var = torch.from_numpy(test_in_var)
    test_in_var = test_in_var.repeat(1, 200, 1)
    test_out_var = test_y
    test_out_var = torch.from_numpy(test_out_var)
    test_dataset = TensorDataset(test_in_var, test_out_var)

    def safe_rename(src, dst):
        if os.path.exists(src) and (not os.path.exists(dst)):
            os.rename(src, dst)

    def maybe_rename_final_images(folder_name, already_done_flag='_renamed_2000.flag'):
        flag_path = os.path.join(folder_name, already_done_flag)
        if os.path.exists(flag_path):
            return
        safe_rename(os.path.join(folder_name, 'train_final.png'), os.path.join(folder_name, 'train_2000.png'))
        safe_rename(os.path.join(folder_name, 'val_final.png'), os.path.join(folder_name, 'val_2000.png'))
        safe_rename(os.path.join(folder_name, 'test_final.png'), os.path.join(folder_name, 'test_2000.png'))
        safe_rename(os.path.join(folder_name, 'bad_final.png'), os.path.join(folder_name, 'bad_2000.png'))
        open(flag_path, 'w').close()

    def should_draw(global_epoch):
        return global_epoch in {2000, 2500, 3000, 3500}
    train_dataloaders = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
    bad_dataloader = DataLoader(bad_dataset, batch_size=config['batch_size'], shuffle=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    rnds = [34, 245325, 5546, 237, 975, 2833, 52057, 73196, 51398, 47032, 99856, 93624, 58974, 56149, 90953, 14830, 24518, 75535, 93641, 93929, 81477, 90468, 38736, 35207, 37220]
    betas = [1.0]
    model_params = {'in_args': {'z': 128, 'hidden_dim': 512, 'bias': False, 'in_feature': 21 + config['embedding_dim'] * 2, 'out_feature': 128}, 'share_args': {'z': 128, 'hidden_dim': 512, 'bias': False, 'in_feature': 128, 'out_feature': 256}, 'b0_args': {'z': 128, 'hidden_dim': 512, 'bias': False, 'in_feature': 256, 'out_feature': 1}, 'b1_args': {'z': 128, 'hidden_dim': 512, 'bias': False, 'in_feature': 128, 'out_feature': 1}, 'd0_args': {'z': 128, 'hidden_dim': 512, 'bias': False, 'in_feature': 256, 'out_feature': 128}, 'd1_args': {'z': 128, 'hidden_dim': 512, 'bias': False, 'in_feature': 128, 'out_feature': 11}, 'd1q_args': {'z': 128, 'hidden_dim': 512, 'bias': False, 'in_feature': 128, 'out_feature': 11}, 'mixer_args': {'s': 256, 'z': 128, 'hidden_dim': 512, 'bias': False, 'n_layers': 5}, 'diser_args': {'z': 128, 'hidden_dim': 512}}
    location = np.float32(np.linspace(0, 1.0, 200, endpoint=False))
    input_encoder = lambda x, a, b: np.concatenate([a * np.sin(2.0 * np.pi * x[..., None] * b), a * np.cos(2.0 * np.pi * x[..., None] * b)], axis=-1) / np.linalg.norm(a)
    bvals = np.float32(np.arange(1, config['embedding_dim'] * config['ed_multiple'] + 1))
    ab_dict = {'{}'.format(p): (bvals ** (-np.float32(p)), bvals) for p in [0.0, 0.5, 1.0, 1.5, 2.0]}
    key = '{}'.format(1.0)
    inputs_ft = input_encoder(location, ab_dict[key][0], ab_dict[key][1])
    inputs_ft = torch.from_numpy(np.expand_dims(inputs_ft, 0)[:, :, ::config['ed_multiple']])
    results_folder_name = os.path.join(config['path_dir'], 'results_second_bad_true_uq_2000_epoch')
    try:
        os.mkdir(results_folder_name)
    except OSError:
        pass
    folder_name = results_folder_name + '/' + '{}_{}_{}_{}_{}'.format(config['model_name'], config['ed_multiple'], config['embedding_dim'], betas[int(config['beta_idx'])], config['rnd'])
    try:
        os.mkdir(folder_name)
    except OSError:
        pass
    check_name = os.path.join(folder_name, 'hyper_final.pth')
    if os.path.exists(check_name):
        hyper = HypyerNet(model_params)
        mixer = Mixer(types.SimpleNamespace(**model_params['mixer_args']))
        diser = Discriminator(types.SimpleNamespace(**model_params['diser_args']))
        hyper = hyper.to(device)
        mixer = mixer.to(device)
        diser = diser.to(device)
        hyper.load_state_dict(torch.load(os.path.join(folder_name, 'hyper_final.pth'), map_location=device))
        mixer.load_state_dict(torch.load(os.path.join(folder_name, 'mixer_final.pth'), map_location=device))
        diser.load_state_dict(torch.load(os.path.join(folder_name, 'diser_final.pth'), map_location=device))
        optimizerQ = torch.optim.Adam(params=list(hyper.parameters()) + list(mixer.parameters()), lr=config['lr'], betas=(0.9, 0.99))
        optimizerD = torch.optim.Adam(diser.parameters(), lr=config['lr'] * 10, betas=(0.5, 0.9))
        FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

        def d_loss(pred_fake, pred_real):
            real_loss_d = torch.nn.MSELoss()(pred_real, FloatTensor(config['n_models'], 1).fill_(1.0))
            fake_loss_d = torch.nn.MSELoss()(pred_fake, FloatTensor(model_params['mixer_args']['n_layers'], config['n_models'], 1).fill_(0.0))
            loss = (real_loss_d + fake_loss_d) / 2
            return loss
        weights = torch.tensor([betas[int(config['beta_idx'])]] + [1.0] * 11)
        weights = weights.view(1, 1, 1, -1).to(device)
        logvar_min = -10.0
        logvar_max = 5.0

        def nll_loss(y_pred, log_var, y_true, w):
            log_var = log_var.clamp(min=logvar_min, max=logvar_max)
            precision = torch.exp(-log_var)
            return torch.mean(precision * (y_true * w - y_pred * w) ** 2 + log_var)

        def train(epoch, dataloader):
            epoch_loss = 0.0
            hyper.train()
            mixer.train()
            diser.train()
            for bidx, samples in enumerate(dataloader):
                s = torch.randn(config['n_models'], model_params['mixer_args']['s']).to(device)
                z = torch.randn(config['n_models'], model_params['diser_args']['z']).to(device)
                data, target = (Variable(samples[0]).to(device), Variable(samples[1]).to(device))
                b = len(data)
                inputs = torch.cat((inputs_ft.repeat((b, 1, 1)).to(device), data + 0.05 * torch.randn_like(data)), axis=-1)
                codes = mixer(s)
                pred_codes_detached = diser(codes.detach())
                pred_z = diser(z)
                loss_d = d_loss(pred_codes_detached, pred_z)
                optimizerD.zero_grad()
                loss_d.backward()
                optimizerD.step()
                pred_codes = diser(codes)
                output, outputUQ = hyper(codes, inputs)
                loss_mean = output[:, :, :, 0:1]
                loss_cum_mean = torch.cumsum(loss_mean, dim=2)
                all_output = torch.cat((output, loss_cum_mean), dim=-1)
                loss_log_var = outputUQ[:, :, :, 0:1]
                loss_log_var = loss_log_var.clamp(min=logvar_min, max=logvar_max)
                loss_std = torch.exp(0.5 * loss_log_var)
                eps = torch.randn((100, *loss_std.shape), device=loss_std.device, dtype=loss_std.dtype)
                loss_sampled = eps * loss_std.unsqueeze(0) + loss_mean.unsqueeze(0)
                cum_samples = torch.cumsum(loss_sampled, dim=3)
                loss_cum_var = cum_samples.var(dim=0, unbiased=True)
                all_uq = torch.cat((outputUQ, loss_cum_var), dim=-1)
                optimizerQ.zero_grad()
                loss_g = torch.nn.MSELoss()(pred_codes, FloatTensor(model_params['mixer_args']['n_layers'], config['n_models'], 1).fill_(1.0))
                if bad_included == 'True':
                    mask = Variable(samples[2]).to(device)
                    all_mean = torch.mul(all_output, mask.unsqueeze(0).repeat(config['n_models'], 1, 1, 1))
                    all_logvar = torch.mul(all_uq, mask.unsqueeze(0).repeat(config['n_models'], 1, 1, 1))
                    all_target = torch.mul(target, mask).unsqueeze(0).repeat(config['n_models'], 1, 1, 1)
                else:
                    all_mean = all_output
                    all_logvar = all_uq
                    all_target = target.unsqueeze(0).repeat(config['n_models'], 1, 1, 1)
                loss_r = nll_loss(all_mean, all_logvar, all_target, weights)
                loss = loss_g + loss_r
                loss.backward()
                optimizerQ.step()
                epoch_loss += loss_r.detach().cpu().numpy()
            epoch_loss /= len(dataloader)
            print('Train Epoch: {}/{} Loss: {:.4f}'.format(epoch, config['epochs'], epoch_loss))
            if epoch % 500 == 0:
                draw(epoch)
            return epoch_loss

        def test(epoch):
            hyper.eval()
            mixer.eval()
            epoch_loss = 0.0
            for bidx, samples in enumerate(val_dataloader):
                data, target = (Variable(samples[0]).to(device), Variable(samples[1]).to(device))
                b = len(data)
                inputs = torch.cat((inputs_ft.repeat((b, 1, 1)).to(device), data), axis=-1)
                s = torch.randn(config['n_models'], model_params['mixer_args']['s']).to(device)
                codes = mixer(s)
                output, outputUQ = hyper(codes, inputs)
                loss = torch.nn.MSELoss(reduction='none')(output, target[:, :, :-1].unsqueeze(0).repeat(config['n_models'], 1, 1, 1)).mean(dim=(0, 1, 2, 3))
                epoch_loss += loss.detach().cpu().numpy()
            epoch_loss /= len(val_dataloader)
            print('Test Epoch: {}/{} Loss: {:.4f}'.format(epoch, config['epochs'], epoch_loss))
            output = output.detach().cpu().numpy()
            return (output, epoch_loss)

        def pred(data):
            hyper.eval()
            mixer.eval()
            b = len(data)
            inputs = torch.cat((inputs_ft.repeat((b, 1, 1)).to(device), data), axis=-1)
            s = torch.randn(config['n_models'], model_params['mixer_args']['s']).to(device)
            codes = mixer(s)
            output, outputUQ = hyper(codes, inputs)
            loss_mean = output[:, :, :, 0:1]
            loss_cum_mean = torch.cumsum(loss_mean, dim=2)
            all_output = torch.cat((output, loss_cum_mean), dim=-1)
            loss_log_var = outputUQ[:, :, :, 0:1]
            loss_log_var = loss_log_var.clamp(min=logvar_min, max=logvar_max)
            loss_std = torch.exp(0.5 * loss_log_var)
            eps = torch.randn((100, *loss_std.shape), device=loss_std.device, dtype=loss_std.dtype)
            loss_sampled = eps * loss_std.unsqueeze(0) + loss_mean.unsqueeze(0)
            cum_samples = torch.cumsum(loss_sampled, dim=3)
            loss_cum_var = cum_samples.var(dim=0, unbiased=True)
            all_uq = torch.cat((outputUQ, loss_cum_var), dim=-1)
            all_var = torch.exp(all_uq)
            predictive_mean = all_output.mean(dim=0)
            epistemic_var = all_output.var(dim=0, unbiased=True)
            aleatoric_var = all_var.mean(dim=0)
            total_var = epistemic_var + aleatoric_var
            predictive_mean = predictive_mean.detach().cpu().numpy()
            total_var = total_var.detach().cpu().numpy()
            lower_bound = predictive_mean - np.sqrt(total_var)
            upper_bound = predictive_mean + np.sqrt(total_var)
            return (output, predictive_mean, lower_bound, upper_bound)

        def draw(epoch):
            hyper.eval()
            mixer.eval()
            rnd_idx = 0
            batch_size = 10
            data, target = (Variable(train_in_var[rnd_idx:rnd_idx + batch_size]).to(device), Variable(train_out_var[rnd_idx:rnd_idx + batch_size]).to(device))
            output, pred_middle, pred_lower, pred_upper = pred(data)
            x_range = np.arange(200)
            k = 0
            fig, axs = plt.subplots(6, 2, figsize=(12, 15))
            for idx, ax in enumerate(axs.flatten()):
                ax.plot(target[k, :, idx].detach().cpu().numpy(), label='T')
                ax.plot(pred_middle[k, :, idx], label='P')
                ax.fill_between(x_range, pred_upper[k, :, idx], pred_lower[k, :, idx], color='tab:orange', label='confidence', alpha=0.2)
            handles, labels = axs[0, 0].get_legend_handles_labels()
            fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=3)
            plt.tight_layout()
            plt.savefig(os.path.join(folder_name, 'train_{}.png'.format(epoch)), bbox_inches='tight')
            plt.close()
            data, target = (Variable(val_in_var[rnd_idx:rnd_idx + batch_size]).to(device), Variable(val_out_var[rnd_idx:rnd_idx + batch_size]).to(device))
            output, pred_middle, pred_lower, pred_upper = pred(data)
            x_range = np.arange(200)
            k = 0
            fig, axs = plt.subplots(6, 2, figsize=(12, 15))
            for idx, ax in enumerate(axs.flatten()):
                ax.plot(target[k, :, idx].detach().cpu().numpy(), label='T')
                ax.plot(pred_middle[k, :, idx], label='P')
                ax.fill_between(x_range, pred_upper[k, :, idx], pred_lower[k, :, idx], color='tab:orange', label='confidence', alpha=0.2)
            handles, labels = axs[0, 0].get_legend_handles_labels()
            fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=3)
            plt.tight_layout()
            plt.savefig(os.path.join(folder_name, 'val_{}.png'.format(epoch)), bbox_inches='tight')
            plt.close()
            rnd_idx = 74
            data, target = (Variable(test_in_var[rnd_idx:rnd_idx + batch_size]).to(device), Variable(test_out_var[rnd_idx:rnd_idx + batch_size]).to(device))
            output, pred_middle, pred_lower, pred_upper = pred(data)
            x_range = np.arange(200)
            k = 0
            fig, axs = plt.subplots(6, 2, figsize=(12, 15))
            for idx, ax in enumerate(axs.flatten()):
                ax.plot(target[k, :, idx].detach().cpu().numpy(), label='T')
                ax.plot(pred_middle[k, :, idx], label='P')
                ax.fill_between(x_range, pred_upper[k, :, idx], pred_lower[k, :, idx], color='tab:orange', label='confidence', alpha=0.2)
            handles, labels = axs[0, 0].get_legend_handles_labels()
            fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=3)
            plt.tight_layout()
            plt.savefig(os.path.join(folder_name, 'test_{}.png'.format(epoch)), bbox_inches='tight')
            plt.close()
            rnd_idx = 0
            data = Variable(bad_in_var[rnd_idx:rnd_idx + batch_size]).to(device)
            target = Variable(bad_out_var[rnd_idx:rnd_idx + batch_size]).to(device)
            info = Variable(bad_info_var[rnd_idx:rnd_idx + batch_size]).to(device)
            output, pred_middle, pred_lower, pred_upper = pred(data)
            x_range = np.arange(200)
            k = 0
            fig, axs = plt.subplots(6, 2, figsize=(12, 15))
            for idx, ax in enumerate(axs.flatten()):
                ax.plot(target[k, :, idx].detach().cpu().numpy(), label='T')
                ax.plot(pred_middle[k, :, idx], label='P', c='tab:orange')
                ax.fill_between(x_range, pred_upper[k, :, idx], pred_lower[k, :, idx], color='tab:orange', label='confidence', alpha=0.2)
                ax.axvline(x=info[k][0].item(), color='red', linestyle='--', label='Final location')
                ax.axvspan(info[k][0].item(), 200, color='black', alpha=0.2, label='Padded Region')
                if idx == 11:
                    ax.axhline(y=info[k][1].item(), color='blue', linestyle='--', label='Final Cumulative error')
            handles, labels = axs[0, 0].get_legend_handles_labels()
            fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=3)
            plt.tight_layout()
            plt.savefig(os.path.join(folder_name, 'bad_{}.png'.format(epoch)), bbox_inches='tight')
            plt.close()

        def run():
            prev_losses, prev_vlosses = ([], [])
            train_loss_path = os.path.join(folder_name, 'train_loss.pkl')
            val_loss_path = os.path.join(folder_name, 'validation_loss.pkl')
            if os.path.exists(train_loss_path):
                with open(train_loss_path, 'rb') as f:
                    prev_losses = pickle.load(f)
            if os.path.exists(val_loss_path):
                with open(val_loss_path, 'rb') as f:
                    prev_vlosses = pickle.load(f)
            assert len(prev_losses) == len(prev_vlosses), 'train/val loss length mismatch'
            start_epoch = len(prev_losses)
            target_total = start_epoch + config['epochs']
            maybe_rename_final_images(folder_name)
            vloss_min = min(prev_vlosses)
            new_losses, new_vlosses = ([], [])
            for epoch in range(start_epoch, target_total):
                epoch_loss = train(epoch, train_dataloaders)
                vepoch_loss = test(epoch)[-1]
                new_losses.append(epoch_loss)
                new_vlosses.append(vepoch_loss)
                if vepoch_loss < vloss_min:
                    vloss_min = vepoch_loss
                    torch.save(hyper.state_dict(), os.path.join(folder_name, 'hyper_best.pth'))
                    torch.save(mixer.state_dict(), os.path.join(folder_name, 'mixer_best.pth'))
                    torch.save(diser.state_dict(), os.path.join(folder_name, 'diser_best.pth'))
                    draw('best')
            torch.save(hyper.state_dict(), os.path.join(folder_name, 'hyper_final.pth'))
            torch.save(mixer.state_dict(), os.path.join(folder_name, 'mixer_final.pth'))
            torch.save(diser.state_dict(), os.path.join(folder_name, 'diser_final.pth'))
            draw('final')
            losses = prev_losses + new_losses
            vlosses = prev_vlosses + new_vlosses
            return (losses, vlosses)
        reproducibility(rnds[int(config['rnd'])])
        loss_dic, vloss_dic = run()
        with open(os.path.join(folder_name, 'train_loss.pkl'), 'wb') as f:
            pickle.dump(loss_dic, f)
        with open(os.path.join(folder_name, 'validation_loss.pkl'), 'wb') as f:
            pickle.dump(vloss_dic, f)
if __name__ == '__main__':
    main()
