import os
import csv
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import pandas as pd
import numpy as np
import glob
import gym
import gc
from tqdm import tqdm
from typing import Tuple, List
from f110_gym.envs.base_classes import Integrator
import f110_gym.envs.f110_env as f110_env
from model import End2Race
from latticeplanner.lattice_planner import LatticePlanner, obsDict2oppoArray
from latticeplanner.utils import project_point_to_centerline, load_config, get_map_paths
from utils import load_raceline_with_speed, calculate_metrics


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='DAgger training for End2Race speed-conditioned model')

    # Data and model paths
    parser.add_argument("--data_path", type=str, default="Dataset_Austin/success",
                        help="Path to initial BC demonstration data")
    parser.add_argument("--model_path", type=str, default="end2race_dagger2.pth",
                        help="Path to save/load model weights")
    parser.add_argument("--dagger_data_dir", type=str, default="DAgger_Data",
                        help="Directory to store aggregated DAgger rollout CSVs")

    # Model configuration
    parser.add_argument("--hidden_scale", type=int, default=4)
    parser.add_argument("--mask_prob", type=float, default=0.1)

    # DAgger configuration
    parser.add_argument("--dagger_iterations", type=int, default=10,
                        help="Number of DAgger outer iterations")
    parser.add_argument("--rollout_steps", type=int, default=30000,
                        help="Max env steps per DAgger rollout")
    parser.add_argument("--map_name", type=str, default="Austin")
    parser.add_argument("--raceline", type=str, default="raceline1",
                        help="Raceline file name (without .csv) for the expert planner")
    parser.add_argument("--beta_start", type=float, default=1.0,
                        help="Initial mixing coefficient (1.0 = pure expert)")
    parser.add_argument("--beta_decay", type=float, default=0.5,
                        help="Multiplicative decay applied to beta each DAgger iteration")
    parser.add_argument("--beta_min", type=float, default=0.0,
                        help="Floor for beta (pure student below this)")

    # Training configuration
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--num_epochs", type=int, default=100,
                        help="Supervised epochs per DAgger iteration")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Dataset (identical contract to train.py)
# ---------------------------------------------------------------------------

class SequenceDataset(Dataset):

    def __init__(self, data_path: str, stride: int = 1, device: str = "cuda"):
        self.stride = stride
        self.device = device

        self.lidar_columns = [f"lidar_{i}" for i in range(360)]
        self.action_columns = ["steer", "desired_speed"]

        self.sequence_length = self._determine_sequence_length(data_path)
        self.sequences = []
        self._load_episodes(data_path)

        print(f"Loaded {len(self.sequences)} sequences from {data_path}")

    def _load_episodes(self, data_path: str):
        """Load CSV files and create sequences."""
        csv_files = sorted(glob.glob(os.path.join(data_path, "*.csv")))
        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            required_cols = self.lidar_columns + self.action_columns
            if not all(col in df.columns for col in required_cols):
                continue
            min_length = self.sequence_length + 1
            if len(df) < min_length:
                continue
            lidar_data = df[self.lidar_columns].values.astype(np.float32)
            action_data = df[self.action_columns].values.astype(np.float32)
            self._create_sequences(lidar_data, action_data)

    def _determine_sequence_length(self, data_path: str) -> int:
        """Automatically determine sequence length from the first CSV file."""
        csv_files = sorted(glob.glob(os.path.join(data_path, "*.csv")))
        first_file = csv_files[0]
        df = pd.read_csv(first_file)
        sequence_length = len(df) - 1
        print(f"Sequence_length: {sequence_length}")
        return sequence_length

    def _create_sequences(self, lidar_data: np.ndarray, action_data: np.ndarray):
        """Create sequences with speed conditioning."""
        lidar_valid = lidar_data[1:]
        action_valid = action_data[1:]
        speed_prev = action_data[:-1, 1:2]
        num_samples = len(lidar_valid)

        for end_idx in range(self.sequence_length - 1, num_samples, self.stride):
            start_idx = end_idx - self.sequence_length + 1
            self.sequences.append({
                'lidar': lidar_valid[start_idx:end_idx + 1],
                'speed': speed_prev[start_idx:end_idx + 1],
                'action': action_valid[start_idx:end_idx + 1],
            })

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sequence = self.sequences[idx]
        lidar_tensor  = torch.tensor(sequence['lidar'],  dtype=torch.float32, device=self.device)
        speed_tensor  = torch.tensor(sequence['speed'],  dtype=torch.float32, device=self.device)
        action_tensor = torch.tensor(sequence['action'], dtype=torch.float32, device=self.device)
        return lidar_tensor, speed_tensor, action_tensor


class DAggerEpisodeDataset(Dataset):
    """
    Wraps a list of pre-built (lidar, speed, action) sequence dicts that were
    collected during DAgger rollouts.  Keeps the same tensor contract as
    SequenceDataset so both can be combined with ConcatDataset.
    """

    def __init__(self, sequences: List[dict], device: str = "cuda"):
        self.sequences = sequences
        self.device = device

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        seq = self.sequences[idx]
        lidar_tensor  = torch.tensor(seq['lidar'],  dtype=torch.float32, device=self.device)
        speed_tensor  = torch.tensor(seq['speed'],  dtype=torch.float32, device=self.device)
        action_tensor = torch.tensor(seq['action'], dtype=torch.float32, device=self.device)
        return lidar_tensor, speed_tensor, action_tensor


# ---------------------------------------------------------------------------
# DAgger rollout with real lattice planner expert
# ---------------------------------------------------------------------------

def collect_dagger_rollout(
    model,
    device,
    map_name,
    rollout_steps,
    beta,
    sequence_length,
    dagger_data_dir,
    iteration,
    raceline='raceline1',
    config_path='latticeplanner/lattice_config.yaml',
):
    """
    Roll out the student in the real environment, labelled by the real lattice
    planner expert (identical planner setup to run_lattice_planner.py).

    At each step:
      - expert produces (steer, speed) via planner.tracker.plan  → always recorded as label
      - with probability beta  → expert action is executed
      - with probability 1-beta → student action is executed

    All (lidar, prev_speed, expert_action) tuples are saved to CSV and also
    returned as pre-built sequence dicts ready for DAggerEpisodeDataset.

    Returns:
        sequences (List[dict])  –  list of {'lidar', 'speed', 'action'} windows
        csv_path  (str)         –  path to the saved CSV for this rollout
    """
    raceline_file = f'{map_name}_raceline.csv'

    # ── Real expert planner (identical setup to run_lattice_planner.py) ───────
    config = load_config(config_path)
    map_directory, map_path = get_map_paths(map_name)
    raceline_path = os.path.join(map_directory, f"{raceline}.csv")
    planner = LatticePlanner(config, map_path, raceline_path)
    cost_weights = np.array([0.12, 2.0, 0.3, 0.5])
    planner.set_parameters({'cost_weights': cost_weights, 'traj_v_scale': 1.0})
    tracker_steps = planner.conf.tracker_steps

    env = gym.make(
        "f110-v0",
        map=f"f1tenth_racetracks/{map_name}/{map_name}_map",
        map_ext=".png",
        num_agents=1,
        timestep=0.01,
        integrator=Integrator.RK4,
    )

    start_pose, initial_speed, waypoints = load_raceline_with_speed(map_name, raceline_file, start_idx=0)
    obs, _, done, _ = env.reset(poses=start_pose)

    hidden_size  = model.gru.hidden_size
    hidden_state = torch.zeros((1, 1, hidden_size), device=device)
    prev_speed   = initial_speed * 0.9
    num_features = 360

    lidar_buf  = []   # (360,)  float32
    speed_buf  = []   # scalar  float32  – prev_speed fed to model
    action_buf = []   # (2,)    float32  – expert label [steer, desired_speed]

    step = 0
    sim_time = 0.0
    sample_interval = 0.1
    next_record_time = sample_interval

    while not done and step < rollout_steps:
        no_opp = np.zeros((1, 4), dtype=np.float64)
        no_opp[0] = [9999.0, 9999.0, 0.0, 0.0]
        best_traj = planner.plan(
            obs['poses_x'][0], obs['poses_y'][0], obs['poses_theta'][0],
            no_opp, obs['linear_vels_x'][0],
        )

        tracker_count = 0
        while not done and tracker_count < tracker_steps and step < rollout_steps:
            # ── Expert label from tracker ─────────────────────────────────────
            exp_steer, exp_speed = planner.tracker.plan(
                obs['poses_x'][0], obs['poses_y'][0], obs['poses_theta'][0],
                obs['linear_vels_x'][0], best_traj,
            )
            exp_steer = float(np.clip(exp_steer, -0.52, 0.52))
            exp_speed = float(exp_speed)

            # ── Execute action (beta mixing) ──────────────────────────────────
            # Student inference only needed when recording or executing student
            if np.random.random() >= beta:
                lidar = np.array(obs["scans"][0]).flatten()
                if len(lidar) > num_features:
                    indices = np.linspace(0, len(lidar) - 1, num_features, dtype=int)
                    lidar = lidar[indices]
                with torch.no_grad():
                    lidar_t = torch.tensor(lidar, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
                    speed_t = torch.tensor([[[prev_speed]]], dtype=torch.float32, device=device)
                    action_seq, hidden_state = model(lidar_t, speed_t, hidden_state)
                    action_t = action_seq[:, -1, :]
                    exec_steer = float(np.clip(action_t[0, 0].item(), -0.52, 0.52))
                    exec_speed = float(action_t[0, 1].item())
            else:
                exec_steer, exec_speed = exp_steer, exp_speed

            obs, timestep, done, _ = env.step(np.array([[exec_steer, exec_speed]]))
            sim_time += timestep
            prev_speed = obs['linear_vels_x'][0]

            if obs['collisions'][0]:
                done = True

            # ── Record at 10Hz to match BC data collector ─────────────────────
            while sim_time >= next_record_time:
                lidar_rec = np.array(obs["scans"][0]).flatten()
                if len(lidar_rec) > num_features:
                    indices = np.linspace(0, len(lidar_rec) - 1, num_features, dtype=int)
                    lidar_rec = lidar_rec[indices]
                lidar_buf.append(lidar_rec.copy())
                speed_buf.append(float(prev_speed))
                action_buf.append([exp_steer, exp_speed])
                next_record_time += sample_interval

            tracker_count += 1
            step += 1

    env.close()
    gc.collect()

    # ── Save raw rollout to CSV ───────────────────────────────────────────────
    os.makedirs(dagger_data_dir, exist_ok=True)
    csv_path = os.path.join(dagger_data_dir, f"rollout_iter{iteration:03d}.csv")

    rows = []
    for t in range(len(lidar_buf)):
        row = {"time": t}
        for i, v in enumerate(lidar_buf[t]):
            row[f"lidar_{i}"] = v
        row["steer"]         = action_buf[t][0]
        row["desired_speed"] = action_buf[t][1]
        rows.append(row)

    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"  Rollout saved → {csv_path}  ({len(rows)} steps)")

    # ── Build sequence windows (mirrors SequenceDataset._create_sequences) ────
    lidar_arr  = np.array(lidar_buf,  dtype=np.float32)
    speed_arr  = np.array(speed_buf,  dtype=np.float32).reshape(-1, 1)
    action_arr = np.array(action_buf, dtype=np.float32)

    if len(lidar_arr) < sequence_length + 1:
        print(f"  Warning: rollout too short ({len(lidar_arr)} steps), skipping sequence creation.")
        return [], csv_path

    lidar_valid  = lidar_arr[1:]
    action_valid = action_arr[1:]
    speed_prev   = speed_arr[:-1]   # shape (T-1, 1)

    sequences = []
    num_samples = len(lidar_valid)
    for end_idx in range(sequence_length - 1, num_samples):
        start_idx = end_idx - sequence_length + 1
        sequences.append({
            'lidar':  lidar_valid[start_idx:end_idx + 1],
            'speed':  speed_prev[start_idx:end_idx + 1],
            'action': action_valid[start_idx:end_idx + 1],
        })

    print(f"  Created {len(sequences)} sequences from rollout.")
    return sequences, csv_path


# ---------------------------------------------------------------------------
# Supervised training inner loop (identical to train.py)
# ---------------------------------------------------------------------------

def train(model_path, model, train_loader, criterion, optimizer, scheduler, num_epochs=100):
    """Supervised training loop (identical contract to train.py)."""
    best_loss = float("inf")

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        with tqdm(train_loader, desc=f"  Epoch {epoch + 1}/{num_epochs}") as pbar:
            for lidar_seq, speed_seq, target_actions in pbar:
                optimizer.zero_grad()

                predicted_actions, _ = model(lidar_seq, speed_seq)

                predicted_actions_flat = predicted_actions.view(-1, predicted_actions.shape[-1])
                target_actions_flat    = target_actions.view(-1, target_actions.shape[-1])

                steer_loss = criterion(predicted_actions_flat[:, 0], target_actions_flat[:, 0])
                speed_loss = criterion(predicted_actions_flat[:, 1], target_actions_flat[:, 1])
                loss = steer_loss + speed_loss * 0.05

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss.item()
                pbar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(train_loader)
        print(f"  Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.5f}")

        scheduler.step(avg_loss)
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), model_path)
            print(f"  New best loss: {best_loss:.5f}. Model saved.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_arguments()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── Phase 0: load initial BC dataset ─────────────────────────────────────
    print("\n=== Loading initial demonstration dataset ===")
    bc_dataset = SequenceDataset(
        data_path=args.data_path,
        stride=1,
        device=device,
    )
    sequence_length = bc_dataset.sequence_length

    # ── Build model ───────────────────────────────────────────────────────────
    model = End2Race(mask_prob=args.mask_prob, hidden_scale=args.hidden_scale).to(device)

    if args.model_path and os.path.exists(args.model_path):
        print(f"Loading pretrained weights from {args.model_path}")
        model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=False))

    criterion = nn.MSELoss()

    # Accumulated DAgger sequences across all iterations
    all_dagger_sequences: List[dict] = []

    beta = args.beta_start

    # ── DAgger outer loop ─────────────────────────────────────────────────────
    for dagger_iter in range(args.dagger_iterations):
        print(f"\n{'='*60}")
        print(f"DAgger iteration {dagger_iter + 1}/{args.dagger_iterations}  (beta={beta:.3f})")
        print(f"{'='*60}")

        # ── Step 1: rollout student in the real environment ───────────────────
        print("Rolling out policy in environment…")
        new_sequences, csv_path = collect_dagger_rollout(
            model=model,
            device=device,
            map_name=args.map_name,
            rollout_steps=args.rollout_steps,
            beta=beta,
            sequence_length=sequence_length,
            dagger_data_dir=args.dagger_data_dir,
            iteration=dagger_iter,
            raceline=args.raceline,
        )
        all_dagger_sequences.extend(new_sequences)

        # ── Step 2: build aggregated dataset (BC data + all DAgger data) ──────
        datasets_to_combine = [bc_dataset]
        if all_dagger_sequences:
            dagger_dataset = DAggerEpisodeDataset(all_dagger_sequences, device=device)
            datasets_to_combine.append(dagger_dataset)

        combined_dataset = ConcatDataset(datasets_to_combine)

        dataloader_kwargs = {'pin_memory': False, 'num_workers': 0} if device.type != "cpu" else {}
        train_loader = DataLoader(
            combined_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
            **dataloader_kwargs,
        )

        print(f"Aggregated dataset size: {len(combined_dataset)} sequences")
        print(f"Train batches: {len(train_loader)}")

        # ── Step 3: retrain from current weights on aggregated data ───────────
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=10
        )

        print(f"Training for {args.num_epochs} epochs…")
        train(
            model_path=args.model_path,
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            num_epochs=args.num_epochs,
        )

        # ── Step 4: decay beta ────────────────────────────────────────────────
        beta = max(args.beta_min, beta * args.beta_decay)

    print("\nDAgger training completed successfully!")