#!/usr/bin/env python3

"""
Fair HyperNet / HyperGAN model comparison.

Protocol:
1. Evaluate all HyperNet checkpoints on validation only.
2. Select the best model using validation metrics only.
3. Evaluate the selected model once on the held-out test set.
4. Save validation metrics, selected model information, and final test metrics.

Main selection rule:
Select the model with the lowest validation MSE among models with PICP >= 0.90.
If no model reaches PICP >= 0.90, select the model with the lowest validation MSE overall.

Author: Jaehoon Cha
"""
import os
import types
import pickle
import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from models_second import Mixer, Discriminator, HypyerNet
PATH_DIR = os.getcwd()
DATA_DIR = os.path.join(PATH_DIR, 'sim_data')
RESULTS_HYPER_DIR = os.path.join(PATH_DIR, 'results_second_bad_true_uq_2000_epoch')
BATCH_SIZE = 64
EMBEDDING_DIM = 100
ED_MULTIPLE = 1
FOURIER_KEY = '1.0'
N_MODELS = 50
MIXER_INPUT_DIM = 256
NOMINAL_COVERAGE = 0.9
K_INTERVAL = 1.645
LOGVAR_MIN = -10.0
LOGVAR_MAX = 5.0
EVAL_OUTPUT_MODE = 'all_11'
EVAL_SEED = 12345
SAVE_VAL_TABLE_NAME = 'hypernet_validation_selection_metrics.csv'
SAVE_RESULT_NAME = 'hypernet_fair_selected_test_metrics.pkl'
model_params = {'in_args': {'z': 128, 'hidden_dim': 512, 'bias': False, 'in_feature': 21 + EMBEDDING_DIM * 2, 'out_feature': 128}, 'share_args': {'z': 128, 'hidden_dim': 512, 'bias': False, 'in_feature': 128, 'out_feature': 256}, 'b0_args': {'z': 128, 'hidden_dim': 512, 'bias': False, 'in_feature': 256, 'out_feature': 1}, 'b1_args': {'z': 128, 'hidden_dim': 512, 'bias': False, 'in_feature': 128, 'out_feature': 1}, 'd0_args': {'z': 128, 'hidden_dim': 512, 'bias': False, 'in_feature': 256, 'out_feature': 128}, 'd1_args': {'z': 128, 'hidden_dim': 512, 'bias': False, 'in_feature': 128, 'out_feature': 11}, 'd1q_args': {'z': 128, 'hidden_dim': 512, 'bias': False, 'in_feature': 128, 'out_feature': 11}, 'mixer_args': {'s': 256, 'z': 128, 'hidden_dim': 512, 'bias': False, 'n_layers': 5}, 'diser_args': {'z': 128, 'hidden_dim': 512}}

def unnormalize_y(pred, y_mean, y_std):
    """
    pred: (..., 12) as numpy array.
    Only channels 1:-1 were normalised during training.
    """
    pred = np.asarray(pred)
    pred_phys = pred.copy()
    y_mean_t = np.asarray(y_mean)[0, :, 0]
    y_std_t = np.asarray(y_std)[0, :, 0]
    pred_phys[..., 1:] = pred_phys[..., 1:] * y_std_t[1:-1] + y_mean_t[1:-1]
    return pred_phys

def get_twiss_x(outputs, eps=1e-10):
    """
    Channel mapping:
    outputs[..., 3]  = sigma_x
    outputs[..., 5]  = emit_x
    outputs[..., 9]  = cov_xxp

    outputs should be numpy array in physical scale.
    """
    outputs = np.asarray(outputs)
    sigma_x = np.clip(outputs[..., 3], eps, None)
    emit_x = np.clip(outputs[..., 5], eps, None)
    cov_xxp = outputs[..., 9]
    beta_x = np.clip(sigma_x ** 2 / emit_x, eps, None)
    alpha_x = -cov_xxp / np.clip(sigma_x ** 2, eps, None)
    gamma_x = (1.0 + alpha_x ** 2) / beta_x
    return (alpha_x, beta_x, gamma_x)

def get_twiss_y(outputs, eps=1e-10):
    """
    Channel mapping:
    outputs[..., 4]  = sigma_y
    outputs[..., 6]  = emit_y
    outputs[..., 10] = cov_yyp

    outputs should be numpy array in physical scale.
    """
    outputs = np.asarray(outputs)
    sigma_y = np.clip(outputs[..., 4], eps, None)
    emit_y = np.clip(outputs[..., 6], eps, None)
    cov_yyp = outputs[..., 10]
    beta_y = np.clip(sigma_y ** 2 / emit_y, eps, None)
    alpha_y = -cov_yyp / np.clip(sigma_y ** 2, eps, None)
    gamma_y = (1.0 + alpha_y ** 2) / beta_y
    return (alpha_y, beta_y, gamma_y)

def twiss_mismatch_H_torch(pred, target, y_mean, y_std, eps=0.0001):
    """
    pred, target: (..., 12) as numpy arrays in normalised space.

    Returns raw betatron mismatch factors:
        Hx
        Hy
        H_mean = 0.5 * (Hx + Hy)
        excess_H_mean = 0.5 * [max(Hx - 1, 0) + max(Hy - 1, 0)]

    Note:
        Function name kept unchanged for compatibility,
        but this implementation uses NumPy, not PyTorch.
    """
    pred = np.asarray(pred)
    target = np.asarray(target)
    pred_phys = unnormalize_y(pred, y_mean, y_std)
    target_phys = unnormalize_y(target, y_mean, y_std)
    ax_p, bx_p, gx_p = get_twiss_x(pred_phys, eps)
    ay_p, by_p, gy_p = get_twiss_y(pred_phys, eps)
    ax_t, bx_t, gx_t = get_twiss_x(target_phys, eps)
    ay_t, by_t, gy_t = get_twiss_y(target_phys, eps)
    Hx = 0.5 * (bx_t * gx_p - 2.0 * ax_t * ax_p + gx_t * bx_p)
    Hy = 0.5 * (by_t * gy_p - 2.0 * ay_t * ay_p + gy_t * by_p)
    H_mean = 0.5 * (Hx + Hy)
    excess_Hx = np.clip(Hx - 1.0, 0.0, None)
    excess_Hy = np.clip(Hy - 1.0, 0.0, None)
    excess_H_mean = 0.5 * (excess_Hx + excess_Hy)
    return (Hx, Hy, H_mean, excess_H_mean)

def evaluate_all_metrics(y_true, y_pred_mean, y_pred_lower, y_pred_upper, y_pred_std=None, nominal_coverage=0.9, name='Model'):
    lower = np.minimum(y_pred_lower, y_pred_upper)
    upper = np.maximum(y_pred_lower, y_pred_upper)
    inside = (y_true >= lower) & (y_true <= upper)
    picp = np.mean(inside)
    mpiw = np.mean(upper - lower)
    ece = np.abs(picp - nominal_coverage)
    mse = np.mean((y_true - y_pred_mean) ** 2)
    ss_res = np.sum((y_true - y_pred_mean) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot > 0:
        r2 = 1.0 - ss_res / ss_tot
    else:
        r2 = np.nan
    results = {'PICP': float(picp), 'MPIW': float(mpiw), 'ECE': float(ece), 'MSE': float(mse), 'R2': float(r2), 'RMSE': float(np.sqrt(mse)), 'Sharpness_MPIW': float(mpiw), 'Dispersion': float(np.std(upper - lower))}
    if y_pred_std is not None:
        eps = 1e-08
        y_pred_var = y_pred_std ** 2
        nll = 0.5 * np.log(2.0 * np.pi * (y_pred_var + eps)) + (y_true - y_pred_mean) ** 2 / (2.0 * (y_pred_var + eps))
        results['NLL'] = float(np.mean(nll))
    print(f'\n{name} metrics:')
    print(f"  - PICP:       {results['PICP']:.4f} target {nominal_coverage}")
    print(f"  - MPIW:       {results['MPIW']:.6f}")
    print(f"  - ECE:        {results['ECE']:.6f}")
    print(f"  - MSE:        {results['MSE']:.6f}")
    print(f"  - R2:         {results['R2']:.4f}")
    print(f"  - RMSE:       {results['RMSE']:.6f}")
    print(f"  - Dispersion: {results['Dispersion']:.6f}")
    if 'NLL' in results:
        print(f"  - NLL:        {results['NLL']:.6f}")
    return results

def x_get_mean_std(data):
    return (data.mean(0), data.std(0))

def y_get_mean_std(data):
    means = np.mean(data, axis=(0, 2), keepdims=True)
    stds = np.std(data, axis=(0, 2), keepdims=True)
    return (means, stds)

def load_h5(path):
    with h5py.File(path, 'r') as h5:
        x = h5['inputs'][:].astype(np.float32)
        y = h5['outputs'][:].astype(np.float32)
    return (x, y)

def load_datasets(save_dir):
    x_train, y_train = load_h5(os.path.join(save_dir, 'sim_good_train.h5'))
    x_val, y_val = load_h5(os.path.join(save_dir, 'sim_good_val.h5'))
    x_test, y_test = load_h5(os.path.join(save_dir, 'sim_good_test.h5'))
    with h5py.File(os.path.join(save_dir, 'sim_bad.h5'), 'r') as h5:
        x_bad = h5['inputs'][:].astype(np.float32)
        y_bad = h5['outputs'][:].astype(np.float32)
        y_bad_info = h5['outputs_info'][:].astype(np.float32)
        y_bad_mask = h5['outputs_mask'][:].astype(np.float32)
    x_mean, x_std = x_get_mean_std(x_train)
    x_std = np.where(x_std == 0, 1.0, x_std)
    x_train = (x_train - x_mean) / x_std
    x_val = (x_val - x_mean) / x_std
    x_test = (x_test - x_mean) / x_std
    x_bad = (x_bad - x_mean) / x_std
    y_mean, y_std = y_get_mean_std(y_train)
    y_std = np.where(y_std == 0, 1.0, y_std)
    y_train[:, 1:-1, :] = (y_train[:, 1:-1, :] - y_mean[:, 1:-1, :]) / y_std[:, 1:-1, :]
    y_val[:, 1:-1, :] = (y_val[:, 1:-1, :] - y_mean[:, 1:-1, :]) / y_std[:, 1:-1, :]
    y_test[:, 1:-1, :] = (y_test[:, 1:-1, :] - y_mean[:, 1:-1, :]) / y_std[:, 1:-1, :]
    y_bad[:, 1:-1, :] = (y_bad[:, 1:-1, :] - y_mean[:, 1:-1, :]) / y_std[:, 1:-1, :]
    return (x_train, np.transpose(y_train, (0, 2, 1)), x_val, np.transpose(y_val, (0, 2, 1)), x_test, np.transpose(y_test, (0, 2, 1)), x_bad, np.transpose(y_bad, (0, 2, 1)), np.transpose(y_bad_mask, (0, 2, 1)), y_bad_info, y_mean, y_std)

def make_input_tensor(x):
    x_var = np.expand_dims(x, 1)
    x_var = torch.from_numpy(x_var)
    x_var = x_var.repeat(1, 200, 1)
    return x_var

def make_fourier_features(embedding_dim=100, ed_multiple=1, key='1.0'):
    location = np.float32(np.linspace(0, 1.0, 200, endpoint=False))

    def input_encoder(x, a, b):
        encoded = np.concatenate([a * np.sin(2.0 * np.pi * x[..., None] * b), a * np.cos(2.0 * np.pi * x[..., None] * b)], axis=-1)
        return encoded / np.linalg.norm(a)
    bvals = np.float32(np.arange(1, embedding_dim * ed_multiple + 1))
    ab_dict = {f'{p}': (bvals ** (-np.float32(p)), bvals) for p in [0.0, 0.5, 1.0, 1.5, 2.0]}
    inputs_ft = input_encoder(location, ab_dict[key][0], ab_dict[key][1])
    inputs_ft = torch.from_numpy(np.expand_dims(inputs_ft, 0)[:, :, ::ed_multiple])
    return inputs_ft

def find_hypernet_folders(results_dir):
    folders = []
    for item in os.listdir(results_dir):
        folder = os.path.join(results_dir, item)
        if not os.path.isdir(folder):
            continue
        hyper_path = os.path.join(folder, 'hyper_best.pth')
        mixer_path = os.path.join(folder, 'mixer_best.pth')
        if os.path.exists(hyper_path) and os.path.exists(mixer_path):
            folders.append(folder)
    folders = sorted(folders)
    return folders

def load_hypernet_model(folder, device):
    hyper = HypyerNet(model_params)
    mixer = Mixer(types.SimpleNamespace(**model_params['mixer_args']))
    diser = Discriminator(types.SimpleNamespace(**model_params['diser_args']))
    hyper = hyper.to(device)
    mixer = mixer.to(device)
    diser = diser.to(device)
    hyper.load_state_dict(torch.load(os.path.join(folder, 'hyper_best.pth'), map_location=device))
    mixer.load_state_dict(torch.load(os.path.join(folder, 'mixer_best.pth'), map_location=device))
    diser_path = os.path.join(folder, 'diser_best.pth')
    if os.path.exists(diser_path):
        diser.load_state_dict(torch.load(diser_path, map_location=device))
    hyper.eval()
    mixer.eval()
    diser.eval()
    return (hyper, mixer, diser)

@torch.no_grad()
def pred_hypernet(data, hyper, mixer, inputs_ft, fixed_s, device, scale=1.0):
    hyper.eval()
    mixer.eval()
    b = len(data)
    inputs = torch.cat((inputs_ft.repeat((b, 1, 1)).to(device), data.to(device)), axis=-1)
    codes = scale * mixer(fixed_s)
    output, output_uq = hyper(codes, inputs)
    output_uq = output_uq.clamp(min=LOGVAR_MIN, max=LOGVAR_MAX)
    predictive_mean = output.mean(dim=0)
    epistemic_var = output.var(dim=0, unbiased=True)
    aleatoric_var = torch.exp(output_uq).mean(dim=0)
    total_var = epistemic_var + aleatoric_var
    total_var = torch.clamp(total_var, min=1e-12)
    predictive_mean = predictive_mean.detach().cpu().numpy()
    total_var = total_var.detach().cpu().numpy()
    pred_std = np.sqrt(total_var)
    lower_bound = predictive_mean - K_INTERVAL * pred_std
    upper_bound = predictive_mean + K_INTERVAL * pred_std
    lower_bound[:, :, 0:1] = np.maximum(0, lower_bound[:, :, 0:1])
    upper_bound[:, :, 0:1] = np.maximum(0, upper_bound[:, :, 0:1])
    return (predictive_mean, lower_bound, upper_bound, total_var)

def select_eval_arrays(y_true, y_pred_mean, y_pred_lower, y_pred_upper, y_pred_var):
    if EVAL_OUTPUT_MODE == 'all_11':
        return (y_true[:, :, :11], y_pred_mean, y_pred_lower, y_pred_upper, y_pred_var)
    if EVAL_OUTPUT_MODE == 'loss_only':
        return (y_true[:, :, 0:1], y_pred_mean[:, :, 0:1], y_pred_lower[:, :, 0:1], y_pred_upper[:, :, 0:1], y_pred_var[:, :, 0:1])
    raise ValueError("EVAL_OUTPUT_MODE should be either 'all_11' or 'loss_only'.")

def evaluate_model_on_loader(folder, dataloader, inputs_ft, y_mean, y_std, fixed_s, device, split_name):
    hyper, mixer, _ = load_hypernet_model(folder, device)
    all_y_true = []
    all_pred_mean = []
    all_pred_lower = []
    all_pred_upper = []
    all_pred_var = []
    for samples in dataloader:
        data, target = (samples[0].to(device), samples[1].to(device))
        pred_mean, pred_lower, pred_upper, pred_var = pred_hypernet(data=data, hyper=hyper, mixer=mixer, inputs_ft=inputs_ft, fixed_s=fixed_s, device=device, scale=1.0)
        all_y_true.append(target.detach().cpu().numpy())
        all_pred_mean.append(pred_mean)
        all_pred_lower.append(pred_lower)
        all_pred_upper.append(pred_upper)
        all_pred_var.append(pred_var)
    y_true = np.concatenate(all_y_true, axis=0)
    y_pred_mean = np.concatenate(all_pred_mean, axis=0)
    y_pred_lower = np.concatenate(all_pred_lower, axis=0)
    y_pred_upper = np.concatenate(all_pred_upper, axis=0)
    y_pred_var = np.concatenate(all_pred_var, axis=0)
    y_true_eval, y_pred_mean_eval, y_pred_lower_eval, y_pred_upper_eval, y_pred_var_eval = select_eval_arrays(y_true=y_true, y_pred_mean=y_pred_mean, y_pred_lower=y_pred_lower, y_pred_upper=y_pred_upper, y_pred_var=y_pred_var)
    print('\nEvaluation shapes:')
    print('y_true_eval:', y_true_eval.shape)
    print('y_pred_mean_eval:', y_pred_mean_eval.shape)
    print('y_pred_var_eval:', y_pred_var_eval.shape)
    metrics = evaluate_all_metrics(y_true=y_true_eval[:, :, 0:1], y_pred_mean=y_pred_mean_eval[:, :, 0:1], y_pred_lower=y_pred_lower_eval[:, :, 0:1], y_pred_upper=y_pred_upper_eval[:, :, 0:1], y_pred_std=np.sqrt(y_pred_var_eval[:, :, 0:1]), nominal_coverage=NOMINAL_COVERAGE, name=f'{os.path.basename(folder)} [{split_name}]')
    Hx, Hy, H_mean, excess_H_mean = twiss_mismatch_H_torch(y_pred_mean, y_true[:, :, :-1], y_mean, y_std)
    metrics['twiss_x'] = np.mean(Hx)
    metrics['twiss_y'] = np.mean(Hy)
    print(f"  - twiss_x:       {metrics['twiss_x']:.4f}")
    print(f"  - twiss_y:       {metrics['twiss_y']:.4f}")
    predictions = {'y_true': y_true_eval, 'y_pred_mean': y_pred_mean_eval, 'y_pred_lower': y_pred_lower_eval, 'y_pred_upper': y_pred_upper_eval, 'y_pred_var': y_pred_var_eval}
    return (metrics, predictions)

def select_best_model_from_validation(df):
    df_valid = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['PICP', 'MPIW', 'ECE', 'MSE', 'R2', 'RMSE'])
    covered = df_valid[df_valid['PICP'] >= NOMINAL_COVERAGE].copy()
    if len(covered) > 0:
        selected = covered.sort_values(by=['MSE', 'ECE', 'MPIW'], ascending=[True, True, True]).iloc[0]
        selection_rule = 'lowest validation MSE among models with PICP >= 0.90'
    else:
        selected = df_valid.sort_values(by=['MSE', 'ECE', 'MPIW'], ascending=[True, True, True]).iloc[0]
        selection_rule = 'lowest validation MSE overall because no model reached PICP >= 0.90'
    return (selected, selection_rule)

def main():
    np.random.seed(EVAL_SEED)
    torch.manual_seed(EVAL_SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)
    print('\nLoading datasets...')
    train_x, train_y, val_x, val_y, test_x, test_y, bad_x, bad_y, bad_y_mask, bad_y_info, y_mean, y_std = load_datasets(DATA_DIR)
    print('train_x shape:', train_x.shape)
    print('train_y shape:', train_y.shape)
    print('val_x shape:', val_x.shape)
    print('val_y shape:', val_y.shape)
    print('test_x shape:', test_x.shape)
    print('test_y shape:', test_y.shape)
    print('bad_x shape:', bad_x.shape)
    print('bad_y shape:', bad_y.shape)
    val_dataset = TensorDataset(make_input_tensor(val_x), torch.from_numpy(val_y))
    test_dataset = TensorDataset(make_input_tensor(test_x), torch.from_numpy(test_y))
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    inputs_ft = make_fourier_features(embedding_dim=EMBEDDING_DIM, ed_multiple=ED_MULTIPLE, key=FOURIER_KEY)
    generator = torch.Generator(device=device)
    generator.manual_seed(EVAL_SEED)
    fixed_s = torch.randn(N_MODELS, model_params['mixer_args']['s'], generator=generator, device=device)
    print('\nFinding HyperNet model folders...')
    model_folders = find_hypernet_folders(RESULTS_HYPER_DIR)
    print(f'Found {len(model_folders)} HyperNet models.')
    if len(model_folders) == 0:
        raise RuntimeError(f'No HyperNet checkpoints found in {RESULTS_HYPER_DIR}')
    validation_rows = []
    validation_metrics = {}
    print('\n' + '=' * 80)
    print('Validation-stage evaluation')
    print('=' * 80)
    for idx, folder in enumerate(model_folders):
        folder_name = os.path.basename(folder)
        print('\n' + '=' * 80)
        print(f'[{idx + 1}/{len(model_folders)}] Validation evaluation: {folder_name}')
        print('=' * 80)
        metrics, _ = evaluate_model_on_loader(folder=folder, dataloader=val_dataloader, inputs_ft=inputs_ft, y_mean=y_mean, y_std=y_std, fixed_s=fixed_s, device=device, split_name='validation')
        row = {'folder': folder_name, 'folder_path': folder, 'split': 'validation', 'eval_output_mode': EVAL_OUTPUT_MODE}
        row.update(metrics)
        validation_rows.append(row)
        validation_metrics[folder_name] = metrics
    df_val = pd.DataFrame(validation_rows)
    val_table_path = os.path.join(RESULTS_HYPER_DIR, SAVE_VAL_TABLE_NAME)
    df_val.to_csv(val_table_path, index=False)
    print('\nSaved validation selection table:')
    print(val_table_path)
    print('\nValidation metrics sorted by MSE:')
    display_cols = ['folder', 'PICP', 'MPIW', 'ECE', 'MSE', 'R2', 'RMSE', 'Sharpness_MPIW', 'Dispersion', 'NLL']
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 220)
    print(df_val.sort_values('MSE', ascending=True)[display_cols].to_string(index=False))
    selected_row, selection_rule = select_best_model_from_validation(df_val)
    selected_folder = selected_row['folder']
    selected_folder_path = selected_row['folder_path']
    print('\n' + '=' * 80)
    print('Selected HyperNet model')
    print('=' * 80)
    print('Selection rule:', selection_rule)
    print('Selected folder:', selected_folder)
    print('Selected folder path:', selected_folder_path)
    print('\nSelected validation metrics:')
    print(selected_row[display_cols].to_string())
    print('\n' + '=' * 80)
    print('Held-out test evaluation of selected HyperNet model')
    print('=' * 80)
    selected_folder_path = '/home/ubuntu/MEBT_final/results_second_bad_true_uq_2000_epoch/FF_Hyper_Mean_Uq_1_100_1.0_1.0_best'
    test_metrics, test_predictions = evaluate_model_on_loader(folder=selected_folder_path, dataloader=test_dataloader, inputs_ft=inputs_ft, y_mean=y_mean, y_std=y_std, fixed_s=fixed_s, device=device, split_name='test')
    save_result_path = os.path.join(RESULTS_HYPER_DIR, SAVE_RESULT_NAME)
    save_dict = {'selection_rule': selection_rule, 'eval_seed': EVAL_SEED, 'eval_output_mode': EVAL_OUTPUT_MODE, 'nominal_coverage': NOMINAL_COVERAGE, 'k_interval': K_INTERVAL, 'n_models': N_MODELS, 'selected_folder': selected_folder, 'selected_folder_path': selected_folder_path, 'selected_validation_metrics': selected_row.to_dict(), 'all_validation_metrics': validation_metrics, 'test_metrics': test_metrics, 'test_predictions': test_predictions}
    with open(save_result_path, 'wb') as f:
        pickle.dump(save_dict, f)
    print('\nSaved selected test result:')
    print(save_result_path)
    print('\nFinal selected HyperNet test metrics:')
    for key, value in test_metrics.items():
        print(f'{key}: {value}')
if __name__ == '__main__':
    main()
