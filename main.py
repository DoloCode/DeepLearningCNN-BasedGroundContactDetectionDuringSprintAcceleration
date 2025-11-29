import numpy as np
import pandas as pd
import copy
from collections import defaultdict
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from scipy.spatial.distance import directed_hausdorff
import wandb
import os
torch.cuda.empty_cache()

#### METRICS
def precision_recall(gt, pred, tol=5):
    ### precision: how many predicted events are correct?
    ### recall: how many ground truth events are predicted?
    matched = set()
    tp = 0
    for g in gt:
        for p in pred:
            if abs(g - p) <= tol and p not in matched:
                matched.add(p)
                tp += 1
                break
    fp = len(pred) - tp
    fn = len(gt) - tp
    prec = tp / (tp + fp + 1e-8)
    rec = tp / (tp + fn + 1e-8)
    return prec, rec

def rand_index(gt, pred, total_len=None):
    total_len = total_len or max(gt + pred + [1])
    g = np.zeros(total_len)
    p = np.zeros(total_len)
    for i in range(1, len(gt)):
        g[gt[i-1]:gt[i]] = i
    for i in range(1, len(pred)):
        p[pred[i-1]:pred[i]] = i
    return (g == p).sum() / total_len

def hausdorff(gt, pred):
    if not gt or not pred:
        return np.nan
    gt = np.array(gt).reshape(-1, 1)
    pred = np.array(pred).reshape(-1, 1)
    return max(directed_hausdorff(gt, pred)[0], directed_hausdorff(pred, gt)[0])

#### PREPROCCSEING
def beta_scale_data(data, beta_min=-1.5, beta_max=1.5):
    # only scale the acc, gyr columns
    base_cols = ['proband', 'run', 'position', 'frame', 'binary']
    imu_cols = [col for col in data.columns if col not in base_cols]
    scaler = MinMaxScaler(feature_range=(beta_min, beta_max))
    data[imu_cols] = scaler.fit_transform(data[imu_cols])
    return data

def prepare_data_and_create_windows(data, probanden, runs, features, window_size, stride):
    X_list, y_list, group_keys = [], [], []
    for proband in probanden:
        for run in runs:
            if run == 3 and proband == 1:
                continue
            for position in [0, 1]:
                run_mask = (
                    (data['proband'] == proband) &
                    (data['run'] == run) &
                    (data['position'] == position)
                )

                data_run = data[run_mask].copy()
                data_run_features = data_run[features].copy()

                X_run_scaled = beta_scale_data(data_run_features, -1.5, 1.5).to_numpy()
                y_run = data_run['binary'].to_numpy()

                n_steps = X_run_scaled.shape[0]

                for start in range(0, n_steps - window_size + 1, stride):
                    end = start + window_size
                    
                    X_window = X_run_scaled[start:end]          
                    y_window = y_run[start:end]                 

                    X_window = X_window.T

                    X_list.append(X_window)
                    y_list.append(y_window)
                    group_keys.append(f"{proband}_{run}_{position}")

    X_np = np.stack(X_list)
    y_np = np.stack(y_list)
    g_keys = np.stack(group_keys)
    return X_np, y_np, g_keys

class WindowedDataset():
    def __init__(self, X, y, group_keys):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.group_keys = group_keys                   
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.group_keys[idx]

#### MODEL BUILD
class InceptionModule(nn.Module):
    def __init__(self, num_channels, out_channels):
        super().__init__()
        self.branch1 = nn.Conv1d(num_channels, out_channels, kernel_size=1, padding=0)

        self.branch3 = nn.Sequential(
            nn.Conv1d(num_channels, out_channels, kernel_size=1, padding=0),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        )

        self.branch5 = nn.Sequential(
            nn.Conv1d(num_channels, out_channels, kernel_size=1, padding=0),
            nn.Conv1d(out_channels, out_channels, kernel_size=5, padding=2)
        )

        self.branch7 = nn.Sequential(
            nn.Conv1d(num_channels, out_channels, kernel_size=1, padding=0),
            nn.Conv1d(out_channels, out_channels, kernel_size=7, padding=3)
        )
        self.bn = nn.BatchNorm1d(out_channels * 4)
        self.relu = nn.ReLU()

    def forward(self, x):
        b1 = self.branch1(x)
        b3 = self.branch3(x)
        b5 = self.branch5(x)
        b7 = self.branch7(x)
        out = torch.cat([b1, b3, b5, b7], dim=1)
        out = self.bn(out)
        return self.relu(out)

class InceptionBlock(nn.Module):
    def __init__(self, num_channels, out_channels):
        super().__init__()
        self.inception = InceptionModule(num_channels, out_channels)
        self.residual = nn.Sequential(
            nn.Conv1d(num_channels, out_channels * 4, kernel_size=1),
            nn.BatchNorm1d(out_channels * 4)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.inception(x)
        res = self.residual(x)
        return self.relu(out + res)

class InceptionModel(nn.Module):
    def __init__(self, num_channels, num_blocks=6, out_channels=32):
        super().__init__()

        blocks = []
        for i in range(num_blocks):
            blocks.append(InceptionBlock(
                num_channels if i == 0 else out_channels * 4,
                out_channels
            ))

        self.blocks = nn.Sequential(*blocks) 

        self.classifier = nn.Sequential(
                    nn.Conv1d(out_channels * 4, 1, kernel_size=1),  
                    nn.Sigmoid()                             
                )

    def forward(self, x):
        x = self.blocks(x)
        x = self.classifier(x).squeeze(1)
        return x

#### Model train
def train_one_epoch(model, dataloader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0.0
    for xb, yb, kb in dataloader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        pred = model(xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def train_with_early_stopping(model, train_loader, val_loader, optimizer, loss_fn,
                            epochs, patience, device):
    best_loss = float('inf')
    counter = 0
    best_model_state = None

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        val_loss, preds, trues = evaluate_loss(model, val_loader, loss_fn, device)
        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered.")
                break

    model.load_state_dict(best_model_state)
    return model

def evaluate_loss(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_trues = []

    with torch.no_grad():
        for xb, yb, kb in dataloader: #kb = keys
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            total_loss += loss.item()
            all_preds.append(pred.cpu())
            all_trues.append(yb.cpu())

    avg_loss = total_loss / len(dataloader)
    all_preds = torch.cat(all_preds, dim=0)
    all_trues = torch.cat(all_trues, dim=0)

    return avg_loss, all_preds, all_trues

#### Data Postprocessing 

def combine_preds_truths(train, val, test):
    all_rows = []
    for split_name, stitched_dict in [('train', train), ('val', val), ('test', test)]:
        for group_key, data in stitched_dict.items():
            preds = data['pred']
            trues = data['true']
            length = len(preds)
            for i in range(length):
                all_rows.append({
                    'split': split_name,
                    'group': group_key,
                    'pred': preds[i],
                    'true': trues[i]
                })
    df = pd.DataFrame(all_rows)
    return df

def get_restitched_predictions_and_truths(model, dataset, config, device):
    """
    Run model on all windows in dataset, then stitch predictions and truths
    back together per group (e.g., athlete/run/position).
    
    Returns:
        dict[group_key] = {
            "full_pred": np.array of stitched predictions,
            "full_true": np.array of stitched ground truths,
        }
    """
    # returns dict of window preds/trues per group (not stitched yet)
    group_stats = model_predictions_by_group(model, dataset, config, device)  
    
    window_length = config.window_size
    stride = config.stride
    
    stitched_results = {}
    for group_key, stats in group_stats.items():
        windows_pred = stats['preds'].numpy()
        windows_true = stats['trues'].numpy()
        
        full_pred, full_true = stitch_windows_to_full_sequence(windows_pred, windows_true, window_length, stride)
        
        stitched_results[group_key] = {
            'pred': full_pred,
            'true': full_true
        }
    return stitched_results

def stitch_windows_to_full_sequence(windows_pred, windows_true, window_length, stride):
    """
    Stitch overlapping windows by averaging overlapping predictions.

    Args:
        windows_pred: numpy array, shape (N_windows, window_length)
        windows_true: numpy array, shape (N_windows, window_length)
        window_length: int
        stride: int
    Returns:
        full_pred, full_true: numpy arrays with full sequence length
    """
    n_windows = windows_pred.shape[0]
    full_length = stride * (n_windows - 1) + window_length

    pred_accum = np.zeros(full_length)
    pred_count = np.zeros(full_length)

    true_accum = np.zeros(full_length)
    true_count = np.zeros(full_length)

    for i in range(n_windows):
        start = i * stride
        end = start + window_length
        pred_accum[start:end] += windows_pred[i]
        pred_count[start:end] += 1

        true_accum[start:end] += windows_true[i]
        true_count[start:end] += 1

    # averaging the preds over windows
    full_pred = pred_accum / pred_count
    full_true = true_accum / true_count
    full_true = np.round(full_true).astype(int)

    return full_pred, full_true

#### Model evaluation
def get_transition_metrics(y_pred, y_true, threshold=0.5, tol=0):

    y_pred = y_pred.squeeze(0) 
    y_true = y_true.squeeze(0) 

    y_pred = (y_pred >= threshold).to(torch.int).cpu().numpy()
    y_true = y_true.to(torch.int).cpu().numpy() 

    y_pred_diff = np.diff(y_pred) 
    y_true_diff = np.diff(y_true)
    pred_events = np.where(y_pred_diff != 0)[0].tolist() 
    true_events = np.where(y_true_diff != 0)[0].tolist()

    if len(pred_events) == 0 and len(true_events) == 0:
        pass

    # Metrics based on transitions
    scores = {}
    scores["rand_index"] = rand_index(true_events, pred_events)
    scores["hausdorff"] = hausdorff(true_events, pred_events)
    # Metrics based on events 
    p, r = precision_recall(true_events, pred_events, tol)
    scores["precision"] = p
    scores["recall"] = r

    return scores

def model_predictions_by_group(model, dataset, config, device):
    model.eval()
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, pin_memory=True, num_workers=0)
    group_stats = defaultdict(lambda: {"preds": [], "trues": []})
    with torch.no_grad():
        for xb, yb, group_keys in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            preds = model(xb)
            preds = preds.cpu()
            yb = yb.cpu()

            for i, gkey in enumerate(group_keys):
                group_stats[gkey]["preds"].append(preds[i])
                group_stats[gkey]["trues"].append(yb[i])

    for group in group_stats:
        group_stats[group]["preds"] = torch.stack(group_stats[group]["preds"], dim=0)
        group_stats[group]["trues"] = torch.stack(group_stats[group]["trues"], dim=0)

    return group_stats

def compute_metrics_per_group(stitched_preds_trues, threshold=0.5, tol=0):
    all_metrics = {}
    aggregate = defaultdict(list)

    for group, data in stitched_preds_trues.items():
        y_pred = torch.tensor(data['pred']).unsqueeze(0)  # shape (1, seq_len)
        y_true = torch.tensor(data['true']).unsqueeze(0)  # shape (1, seq_len)

        metrics = get_transition_metrics(y_pred, y_true, threshold=threshold, tol=tol)
        all_metrics[group] = metrics

        for k, v in metrics.items():
            aggregate[k].append(v)

    mean_metrics = {k: np.mean(v) for k, v in aggregate.items()}
    return all_metrics, mean_metrics


## logging
def log_metrics_to_wandb(metrics_per_group, mean_metrics, prefix):
    for metric_name, value in mean_metrics.items():
        wandb.log({f"{prefix}/mean_{metric_name}": value})

    for group, metrics in metrics_per_group.items():
        for metric_name, value in metrics.items():
            wandb.log({f"{prefix}/{group}_{metric_name}": value})

#### TRAIN PIPELINE
def train_model(data):

    run = wandb.init()  
    config = wandb.config
    window_size = config.window_size
    stride = config.stride
    num_features = len(config.feature_col)
    num_blocks = config.num_blocks
    if stride >= window_size:
        print(f"Skipping config: stride ({stride}) >= window_size ({window_size})")
        wandb.finish(exit_code=0) 
        return

    # preprocessing: beta scaling, window generation
    train_x_input, train_y_input, train_keys    = prepare_data_and_create_windows(data, probanden=config.train_athletes, runs=config.train_runs, features = config.feature_col, window_size = window_size, stride = stride)
    val_x_input,   val_y_input,   val_keys      = prepare_data_and_create_windows(data, probanden=config.val_athletes,   runs=config.val_runs, features = config.feature_col, window_size = window_size,   stride = stride)
    test_x_input,  test_y_input,  test_keys     = prepare_data_and_create_windows(data, probanden=config.test_athletes,  runs=config.test_runs, features = config.feature_col, window_size = window_size,  stride = stride)

    train_dataset   = WindowedDataset(train_x_input, train_y_input, train_keys)
    val_dataset     = WindowedDataset(val_x_input, val_y_input,     val_keys )
    test_dataset    = WindowedDataset(test_x_input, test_y_input,   test_keys)

    ## creating the general model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = InceptionModel(num_features, num_classes=2, num_blocks = num_blocks)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    loss_fn = nn.BCELoss()

    train_loader = DataLoader(train_dataset,   batch_size=config.batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,     batch_size=config.batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset,    batch_size=config.batch_size, shuffle=False)

    trained_model_general = train_with_early_stopping(model, train_loader, val_loader, optimizer, loss_fn,
                                                      epochs=config.epochs, patience=5, device=device)

    ## metrics & logging
    # generate model prediction + ground truth in the data structure per run (dewindowing) (per group)
    train_stitched_preds_trues = get_restitched_predictions_and_truths(trained_model_general, train_dataset, config, device)
    val_stitched_preds_trues   = get_restitched_predictions_and_truths(trained_model_general, val_dataset, config, device)
    test_stitched_preds_trues  = get_restitched_predictions_and_truths(trained_model_general, test_dataset, config, device)
    all_preds_trues            = combine_preds_truths(train_stitched_preds_trues, val_stitched_preds_trues, test_stitched_preds_trues)

    train_metrics,  train_mean_metrics   = compute_metrics_per_group(train_stitched_preds_trues)
    val_metrics,    val_mean_metrics     = compute_metrics_per_group(val_stitched_preds_trues)
    test_metrics,   test_mean_metrics    = compute_metrics_per_group(test_stitched_preds_trues)

    log_metrics_to_wandb(train_metrics, train_mean_metrics, prefix="train")
    log_metrics_to_wandb(val_metrics, val_mean_metrics, prefix="val")
    log_metrics_to_wandb(test_metrics, test_mean_metrics, prefix="test")

    wandb.finish()
    torch.cuda.empty_cache()

#### Sweep config
sweep_config = {
    'method': 'bayes',
    'metric': {'name': 'val/mean_hausdorff', 'goal': 'minimize'},
    'parameters': {
        # define your data split according to your data
        'train_athletes': {'values': [[2, 3, 4, 5, 7, 8, 9, 10, 11, 12]]},
        'train_runs':     {'values': [[1, 2]]}, 
        'val_athletes':   {'values': [[2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]},
        'val_runs':       {'values': [[3]]}, 
        'test_athletes':  {'values': [[1, 12]]},
        'test_runs':      {'values': [[1, 2, 3]]},

        'feature_col':    {'values': [['ares', 'gres']]},
        'window_size':    {'values': [25, 50, 75, 100, 150, 200]},
        'stride':         {'values': [5, 10, 15, 20, 25, 30]},
        'num_blocks':     {'values': [6, 9, 12, 15, 18, 21]},
        'lr':             {'values': [1e-2, 1e-3, 1e-4]},
        'batch_size':     {'values': [50]}, 
        'epochs':         {'values': [30]},
    }
}

#### MAINN
data = "Insert your path to the data"

## RUN A SWEEP
sweep_id = wandb.sweep(sweep_config, project="1DCNN Inception - yourname")
wandb.agent(sweep_id, function=lambda: train_model(data.copy()), count=500)
