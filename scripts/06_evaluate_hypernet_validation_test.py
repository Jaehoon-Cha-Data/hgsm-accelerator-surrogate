import os
import types

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from models_second import HypyerNet, Mixer

PATH_DIR = os.getcwd()
DATA_DIR = os.path.join(PATH_DIR, "sim_data")
CHECKPOINT_DIR = os.path.join(PATH_DIR, "check_points")
OUTPUT_PATH = os.path.join(CHECKPOINT_DIR, "inference_results.npz")

BATCH_SIZE = 64
EMBEDDING_DIM = 100
ED_MULTIPLE = 1
FOURIER_KEY = "1.0"
N_MODELS = 50
EVAL_SEED = 12345
K_INTERVAL = 1.645
LOGVAR_MIN = -10.0
LOGVAR_MAX = 5.0

MODEL_PARAMS = {
    "in_args": {
        "z": 128,
        "hidden_dim": 512,
        "bias": False,
        "in_feature": 21 + EMBEDDING_DIM * 2,
        "out_feature": 128,
    },
    "share_args": {
        "z": 128,
        "hidden_dim": 512,
        "bias": False,
        "in_feature": 128,
        "out_feature": 256,
    },
    "d0_args": {
        "z": 128,
        "hidden_dim": 512,
        "bias": False,
        "in_feature": 256,
        "out_feature": 128,
    },
    "d1_args": {
        "z": 128,
        "hidden_dim": 512,
        "bias": False,
        "in_feature": 128,
        "out_feature": 11,
    },
    "d1q_args": {
        "z": 128,
        "hidden_dim": 512,
        "bias": False,
        "in_feature": 128,
        "out_feature": 11,
    },
    "mixer_args": {
        "s": 256,
        "z": 128,
        "hidden_dim": 512,
        "bias": False,
        "n_layers": 5,
    },
}


def load_h5(path):
    with h5py.File(path, "r") as h5:
        inputs = h5["inputs"][:].astype(np.float32)
        outputs = h5["outputs"][:].astype(np.float32)
    return inputs, outputs


def load_test_data():
    train_x, train_y = load_h5(os.path.join(DATA_DIR, "sim_good_train.h5"))
    test_x, test_y = load_h5(os.path.join(DATA_DIR, "sim_good_test.h5"))

    x_mean = train_x.mean(axis=0)
    x_std = train_x.std(axis=0)
    x_std = np.where(x_std == 0.0, 1.0, x_std)

    y_mean = np.mean(train_y, axis=(0, 2), keepdims=True)
    y_std = np.std(train_y, axis=(0, 2), keepdims=True)
    y_std = np.where(y_std == 0.0, 1.0, y_std)

    test_x = (test_x - x_mean) / x_std
    test_y[:, 1:-1, :] = (
        test_y[:, 1:-1, :] - y_mean[:, 1:-1, :]
    ) / y_std[:, 1:-1, :]

    test_x = torch.from_numpy(test_x[:, None, :]).repeat(1, 200, 1)
    test_y = torch.from_numpy(np.transpose(test_y, (0, 2, 1)))

    return TensorDataset(test_x, test_y)


def make_fourier_features():
    location = np.linspace(
        0.0,
        1.0,
        200,
        endpoint=False,
        dtype=np.float32,
    )
    frequencies = np.arange(
        1,
        EMBEDDING_DIM * ED_MULTIPLE + 1,
        dtype=np.float32,
    )
    amplitudes = frequencies ** -np.float32(FOURIER_KEY)

    encoded = np.concatenate(
        [
            amplitudes
            * np.sin(2.0 * np.pi * location[:, None] * frequencies),
            amplitudes
            * np.cos(2.0 * np.pi * location[:, None] * frequencies),
        ],
        axis=-1,
    )
    encoded /= np.linalg.norm(amplitudes)

    return torch.from_numpy(encoded[None, :, ::ED_MULTIPLE])


def load_model(device):
    hyper_path = os.path.join(CHECKPOINT_DIR, "hyper_best.pth")
    mixer_path = os.path.join(CHECKPOINT_DIR, "mixer_best.pth")

    if not os.path.exists(hyper_path):
        raise FileNotFoundError(hyper_path)

    if not os.path.exists(mixer_path):
        raise FileNotFoundError(mixer_path)

    hyper = HypyerNet(MODEL_PARAMS).to(device)
    mixer = Mixer(
        types.SimpleNamespace(**MODEL_PARAMS["mixer_args"])
    ).to(device)

    hyper.load_state_dict(
        torch.load(hyper_path, map_location=device)
    )
    mixer.load_state_dict(
        torch.load(mixer_path, map_location=device)
    )

    hyper.eval()
    mixer.eval()

    return hyper, mixer


@torch.no_grad()
def run_inference(dataloader, hyper, mixer, fourier_features, device):
    generator = torch.Generator(device=device)
    generator.manual_seed(EVAL_SEED)

    fixed_latent = torch.randn(
        N_MODELS,
        MODEL_PARAMS["mixer_args"]["s"],
        generator=generator,
        device=device,
    )
    codes = mixer(fixed_latent)

    targets = []
    means = []
    lower_bounds = []
    upper_bounds = []
    standard_deviations = []

    for inputs, target in dataloader:
        inputs = inputs.to(device)

        fourier = fourier_features.repeat(
            inputs.shape[0],
            1,
            1,
        ).to(device)
        model_inputs = torch.cat([fourier, inputs], dim=-1)

        output, output_log_variance = hyper(codes, model_inputs)
        output_log_variance = output_log_variance.clamp(
            min=LOGVAR_MIN,
            max=LOGVAR_MAX,
        )

        predictive_mean = output.mean(dim=0)
        epistemic_variance = output.var(dim=0, unbiased=True)
        aleatoric_variance = torch.exp(
            output_log_variance
        ).mean(dim=0)

        total_variance = torch.clamp(
            epistemic_variance + aleatoric_variance,
            min=1e-12,
        )
        predictive_std = torch.sqrt(total_variance)

        lower = predictive_mean - K_INTERVAL * predictive_std
        upper = predictive_mean + K_INTERVAL * predictive_std

        lower[..., 0] = torch.clamp(lower[..., 0], min=0.0)
        upper[..., 0] = torch.clamp(upper[..., 0], min=0.0)

        targets.append(target[..., :11].numpy())
        means.append(predictive_mean.cpu().numpy())
        lower_bounds.append(lower.cpu().numpy())
        upper_bounds.append(upper.cpu().numpy())
        standard_deviations.append(predictive_std.cpu().numpy())

    return {
        "target": np.concatenate(targets),
        "mean": np.concatenate(means),
        "lower": np.concatenate(lower_bounds),
        "upper": np.concatenate(upper_bounds),
        "std": np.concatenate(standard_deviations),
    }


def print_metrics(results):
    target = results["target"]
    mean = results["mean"]
    lower = results["lower"]
    upper = results["upper"]

    mse = np.mean((target - mean) ** 2)
    rmse = np.sqrt(mse)
    coverage = np.mean((target >= lower) & (target <= upper))
    interval_width = np.mean(upper - lower)

    loss_mse = np.mean(
        (target[..., 0] - mean[..., 0]) ** 2
    )

    print(f"Samples: {target.shape[0]}")
    print(f"Target shape: {target.shape}")
    print(f"Prediction shape: {mean.shape}")
    print(f"MSE: {mse:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"Loss-channel MSE: {loss_mse:.6f}")
    print(f"Coverage: {coverage:.4f}")
    print(f"Mean interval width: {interval_width:.6f}")


def main():
    np.random.seed(EVAL_SEED)
    torch.manual_seed(EVAL_SEED)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")
    print(f"Checkpoint directory: {CHECKPOINT_DIR}")

    dataset = load_test_data()
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    fourier_features = make_fourier_features()
    hyper, mixer = load_model(device)

    results = run_inference(
        dataloader,
        hyper,
        mixer,
        fourier_features,
        device,
    )

    print_metrics(results)
    np.savez_compressed(OUTPUT_PATH, **results)
    print(f"Saved inference results: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
