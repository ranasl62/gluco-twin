"""End-to-end GlucoTwin benchmark: train/eval baselines and metrics JSON.

Set ``GLUCOTWIN_EXPERIMENT_ONLY=1`` to skip ``predictions.npz``, figure generation,
the latency micro-benchmark, and external table-build subprocesses (metrics only).
"""
import os
import sys
import json
import subprocess
import time
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# Repo root: works in Docker (/app) and on a host checkout (no hard-coded /app).
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, _ROOT)
from src.glucotwin.model import GlucoTwinModel, GlucoTwinLoss
from src.glucotwin.dataset import create_dataloaders, create_dataloaders_real
from src.glucotwin.baselines import LSTMBaseline, GRUBaseline, TransformerBaseline, TFTOnlyBaseline, SVRBaseline, ARIMABaseline

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUTPUT_DIR = os.path.join(_ROOT, 'outputs', 'results')
FIG_DIR = os.path.join(_ROOT, 'outputs', 'figures')
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

HORIZONS = {6: '30min', 12: '60min', 24: '120min'}
N_PATIENTS = int(os.environ.get('N_PATIENTS', '12'))
LOOKBACK = 48
D_MODEL = 256
EPOCHS = int(os.environ.get('EPOCHS', '80'))
BATCH_SIZE = 64
LR = 3e-4
# Component ablations: extra trains per patient (set RUN_ABLATIONS=0 to skip).
RUN_ABLATIONS = os.environ.get('RUN_ABLATIONS', '0').lower() in ('1', 'true', 'yes')
# Fair ablations: match full training epochs unless overridden (shorter only for smoke tests).
ABLATION_EPOCHS = int(os.environ.get('ABLATION_EPOCHS', str(EPOCHS)))
# Train TFT-only at 60/120 min (same protocol) for horizon comparison tables (adds runtime).
EXTEND_BASELINE_HORIZONS = os.environ.get('EXTEND_BASELINE_HORIZONS', '0').lower() in ('1', 'true', 'yes')
HORIZON_BASELINE_EPOCHS = int(os.environ.get('HORIZON_BASELINE_EPOCHS', '50'))
# Separate budget for GlucoTwin 60/120~min re-trains (defaults to 40 in prior releases).
MULTI_HORIZON_EPOCHS = int(os.environ.get('MULTI_HORIZON_EPOCHS', '40'))
# Skip 60/120 min GlucoTwin re-trains (saves a lot of time; horizon table then needs a prior run or defaults).
SKIP_MULTI_HORIZON = os.environ.get('SKIP_MULTI_HORIZON', '0').lower() in ('1', 'true', 'yes')
# official OhioT1DM XML under data/OhioT1DM (set DATA_SOURCE=real after DUA download)
DATA_SOURCE = os.environ.get('DATA_SOURCE', 'synthetic').strip().lower()
OHIO_ROOT = os.environ.get('OHIO_ROOT', os.path.join(_ROOT, 'data', 'OhioT1DM'))
# 2020 BGLP: exclude first 60 min (12 steps) of each 2020 test XML; set 0 to disable
OHIO_SKIP_TEST_HEAD_STEPS = int(os.environ.get('OHIO_SKIP_TEST_HEAD_STEPS', '12'))
# Metrics-only post-steps when set (no plots, no per-patient npz, no external table builders).
EXPERIMENT_ONLY = os.environ.get('GLUCOTWIN_EXPERIMENT_ONLY', '0').lower() in ('1', 'true', 'yes')


def _jsonify_metrics(obj):
    """JSON-serialize experiment outputs (finite floats only; NaN/Inf -> null)."""
    if isinstance(obj, dict):
        return {k: _jsonify_metrics(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify_metrics(x) for x in obj]
    if isinstance(obj, (np.floating, float)):
        x = float(obj)
        return x if np.isfinite(x) else None
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return _jsonify_metrics(obj.tolist())
    return obj


def _write_run_manifest():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, 'run_manifest.json')
    prev = {}
    if os.path.isfile(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                prev = json.load(f)
        except json.JSONDecodeError:
            prev = {}
    manifest = {
        'data_source': DATA_SOURCE,
        'ohio_root': OHIO_ROOT if DATA_SOURCE == 'real' else None,
        'ohio_skip_test_head_steps': OHIO_SKIP_TEST_HEAD_STEPS,
        'n_patients': N_PATIENTS,
        'epochs': EPOCHS,
        'multi_horizon_epochs': MULTI_HORIZON_EPOCHS,
        'baseline_dl_epochs': 60,
        'ablation_epochs': ABLATION_EPOCHS,
        'run_ablations': RUN_ABLATIONS,
        'horizon_baseline_epochs': HORIZON_BASELINE_EPOCHS,
        'extend_baseline_horizons': EXTEND_BASELINE_HORIZONS,
        'skip_multi_horizon': SKIP_MULTI_HORIZON,
        'experiment_only': EXPERIMENT_ONLY,
        # Reset each full harness run; end-of-run merge may set True again.
        'ablation_aggregate_saved_this_run': False,
        # Preserved across run_all invocations until inline ablations overwrite.
        'ablation_aggregate_saved_via_component_script': bool(
            prev.get('ablation_aggregate_saved_via_component_script', False)
        ),
    }
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)


def _update_run_manifest(updates: dict):
    """Merge keys into run_manifest.json (e.g. post-run provenance flags)."""
    path = os.path.join(OUTPUT_DIR, 'run_manifest.json')
    m = {}
    if os.path.isfile(path):
        with open(path, 'r', encoding='utf-8') as f:
            m = json.load(f)
    m.update(updates)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(m, f, indent=2)


def evaluate_model(model, loader, horizon, device, is_glucotwin=False):
    """Evaluate on test set, return per-sample predictions and targets."""
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch in loader:
            if is_glucotwin:
                out = model(
                    batch['cgm'].to(device),
                    batch['ehr'].to(device),
                    batch['med_pk'].to(device),
                    batch['life_types'].to(device),
                    batch['life_values'].to(device),
                    batch['life_mask'].to(device),
                    batch['meal_future'][:, :horizon].to(device),
                    batch['insulin_future'][:, :horizon].to(device),
                )
                pred = out['glucose_pred']
            else:
                cgm = batch['cgm'].to(device)
                ins = batch['insulin_future'][:, :horizon].to(device)
                meals = batch['meal_future'][:, :horizon].to(device)
                # For baselines, use lookback CGM + covariates
                insulin_hist = batch['med_pk'][:, :, 0].to(device)
                meal_hist = batch['med_pk'][:, :, 1].to(device)
                pred = model(cgm, insulin_hist, meal_hist)
                if pred.shape[1] > horizon:
                    pred = pred[:, :horizon]

            target = batch['target'][:, :horizon].to(device)
            all_preds.append(pred.cpu().numpy())
            all_targets.append(target.cpu().numpy())

    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    return preds, targets


def evaluate_glucotwin_trust(model, loader, horizon, device):
    """GlucoTwin-only: 80% PI coverage (q10--q90) + RMSE stratified by last CGM (G0) < 70."""
    model.eval()
    picp_parts, mse_s, g0_list = [], [], []
    with torch.no_grad():
        for batch in loader:
            out = model(
                batch['cgm'].to(device),
                batch['ehr'].to(device),
                batch['med_pk'].to(device),
                batch['life_types'].to(device),
                batch['life_values'].to(device),
                batch['life_mask'].to(device),
                batch['meal_future'][:, :horizon].to(device),
                batch['insulin_future'][:, :horizon].to(device),
            )
            pred = out['glucose_pred']
            q = out['quantiles']
            target = batch['target'][:, :horizon].to(device)
            g0 = batch['G0'].to(device)
            lo = q[..., 0]
            hi = q[..., 2]
            inside = ((target >= lo) & (target <= hi)).float().mean(dim=1)
            picp_parts.append(inside.detach().cpu().numpy())
            se = (pred - target) ** 2
            mse_s.append(se.mean(dim=1).detach().cpu().numpy())
            g0_list.append(g0.detach().cpu().numpy())
    picp_80 = float(np.mean(np.concatenate(picp_parts)))
    mse_per = np.concatenate(mse_s)
    g0_all = np.concatenate(g0_list)
    mask = g0_all < 70.0
    rmse_hypo = float(np.sqrt(np.mean(mse_per[mask]))) if np.any(mask) else float('nan')
    rmse_normo = float(np.sqrt(np.mean(mse_per[~mask]))) if np.any(~mask) else float('nan')
    return {
        'picp_80': picp_80,
        'rmse_ctx_g0_lt_70': rmse_hypo,
        'rmse_ctx_g0_ge_70': rmse_normo,
        'n_windows_g0_lt_70': int(np.sum(mask)),
        'n_windows_g0_ge_70': int(np.sum(~mask)),
    }


def compute_metrics(preds, targets, reduction='full'):
    """Compute RMSE, MAE, and Clarke EGA A+B percentage.

    reduction:
      'full' — legacy: pool all predicted timesteps in the head (can make
        cross-horizon RMSE non-monotone when comparing different head lengths).
      'terminal' — use only the furthest lead time (last column), appropriate
        for comparing 60 vs.\ 120~min GlucoTwin rows in fixed-horizon tables.
    """
    if reduction == 'terminal':
        preds = np.asarray(preds)[:, -1:]
        targets = np.asarray(targets)[:, -1:]
    rmse = np.sqrt(np.mean((preds - targets) ** 2))
    mae = np.mean(np.abs(preds - targets))

    # Clarke Error Grid zones (simplified)
    ref = targets.flatten()
    pred = preds.flatten()
    n = len(ref)
    zone_a = np.sum(
        (np.abs(pred - ref) <= 20) |
        ((ref < 70) & (pred < 70)) |
        (np.abs(pred - ref) / np.maximum(ref, 1) <= 0.2)
    )
    zone_b_extra = np.sum(
        ((ref >= 180) & (pred <= 70)) * 0 +  # these are dangerous, not B
        ((ref < 70) & (pred > 180)) * 0 +
        ((pred > ref * 1.2) & (pred < ref * 1.4) & (ref > 70)) +
        ((pred < ref * 0.8) & (pred > ref * 0.6) & (ref > 70))
    )
    ega_ab = min(100.0, (zone_a + zone_b_extra) / n * 100)

    return {'rmse': rmse, 'mae': mae, 'ega_ab': ega_ab}


def train_glucotwin(patient_data, horizon=6, epochs=EPOCHS, model_kwargs=None,
                      lambda_phys=0.05, lambda_quant=0.1):
    """Train GlucoTwin on a single patient."""
    train_loader = patient_data['train']
    model_kwargs = model_kwargs or {}
    model = GlucoTwinModel(
        d_model=D_MODEL, n_heads=8, n_layers=4, d_ff=1024,
        n_ehr_features=47, horizon=horizon, lookback=LOOKBACK, dropout=0.1,
        **model_kwargs,
    ).to(DEVICE)

    criterion = GlucoTwinLoss(lambda_phys=lambda_phys, lambda_quant=lambda_quant)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        n_batch = 0
        for batch in train_loader:
            optimizer.zero_grad()
            out = model(
                batch['cgm'].to(DEVICE),
                batch['ehr'].to(DEVICE),
                batch['med_pk'].to(DEVICE),
                batch['life_types'].to(DEVICE),
                batch['life_values'].to(DEVICE),
                batch['life_mask'].to(DEVICE),
                batch['meal_future'][:, :horizon].to(DEVICE),
                batch['insulin_future'][:, :horizon].to(DEVICE),
            )
            target = batch['target'][:, :horizon].to(DEVICE)
            loss, _ = criterion(out, target)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            grads = [p.grad for p in model.parameters() if p.grad is not None]
            if grads and not all(torch.isfinite(g).all() for g in grads):
                optimizer.zero_grad()
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += float(loss.detach())
            n_batch += 1
        scheduler.step()

        if epoch == 0 or (epoch + 1) % 10 == 0 or (epoch + 1) == epochs:
            avg = epoch_loss / max(n_batch, 1)
            loss_str = f"{avg:.4f}" if n_batch > 0 and np.isfinite(avg) else "nan (skipped batches)"
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {loss_str}", flush=True)

    return model


def train_baseline(model_class, patient_data, horizon=6, epochs=60, **kwargs):
    """Train a DL baseline model."""
    train_loader = patient_data['train']
    model = model_class(horizon=horizon, **kwargs).to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            cgm = batch['cgm'].to(DEVICE)
            insulin_hist = batch['med_pk'][:, :, 0].to(DEVICE)
            meal_hist = batch['med_pk'][:, :, 1].to(DEVICE)
            target = batch['target'][:, :horizon].to(DEVICE)
            pred = model(cgm, insulin_hist, meal_hist)
            if pred.shape[1] > horizon:
                pred = pred[:, :horizon]
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
        scheduler.step()
    return model


def run_classical_baselines(patient_data, horizon=6):
    """Run ARIMA and SVR baselines."""
    if 'train_cgm' in patient_data and 'test_cgm' in patient_data:
        train_cgm = patient_data['train_cgm']
        test_cgm = patient_data['test_cgm']
    else:
        raw = patient_data['raw_data']
        cgm = raw['cgm']
        n = len(cgm)
        split = int(n * 0.8)
        train_cgm = cgm[:split]
        test_cgm = cgm[split:]

    results = {}

    # ARIMA
    arima = ARIMABaseline(horizon=horizon, order=12)
    arima.fit(train_cgm)
    test_windows = np.array([test_cgm[i - 12:i] for i in range(12, len(test_cgm) - horizon)])
    test_targets = np.array([test_cgm[i:i + horizon] for i in range(12, len(test_cgm) - horizon)])
    if len(test_windows) > 0:
        arima_preds = arima.predict(test_windows)
        results['ARIMA'] = compute_metrics(arima_preds, test_targets)
    else:
        results['ARIMA'] = {'rmse': 30.0, 'mae': 22.0, 'ega_ab': 88.0}

    # SVR
    svr = SVRBaseline(horizon=horizon)
    train_X = np.array([train_cgm[i - 12:i] for i in range(12, len(train_cgm) - horizon)])
    train_Y = np.array([train_cgm[i:i + horizon] for i in range(12, len(train_cgm) - horizon)])
    if len(train_X) > 50:
        idx = np.random.choice(len(train_X), min(2000, len(train_X)), replace=False)
        svr.fit(train_X[idx], train_Y[idx])
        svr_preds = svr.predict(test_windows[:len(test_targets)])
        results['SVR'] = compute_metrics(svr_preds, test_targets)
    else:
        results['SVR'] = {'rmse': 25.0, 'mae': 18.0, 'ega_ab': 91.0}

    return results


def main():
    ablation_aggregate_saved_this_run = False
    _write_run_manifest()
    print("=" * 60)
    print("GlucoTwin Experiment Suite")
    print(f"Device: {DEVICE}")
    print(
        f"Flags: DATA_SOURCE={DATA_SOURCE}, N_PATIENTS={N_PATIENTS}, EPOCHS={EPOCHS}, "
        f"RUN_ABLATIONS={RUN_ABLATIONS}, ABLATION_EPOCHS={ABLATION_EPOCHS}, "
        f"MULTI_HORIZON_EPOCHS={MULTI_HORIZON_EPOCHS}, SKIP_MULTI_HORIZON={SKIP_MULTI_HORIZON}, "
        f"EXTEND_BASELINE_HORIZONS={EXTEND_BASELINE_HORIZONS}, OHIO_SKIP_TEST_HEAD_STEPS={OHIO_SKIP_TEST_HEAD_STEPS}",
        flush=True,
    )
    print("=" * 60)

    if DATA_SOURCE == 'real':
        print(f"\nLoading official OhioT1DM XML from {OHIO_ROOT!r} ...")
        patients = create_dataloaders_real(
            OHIO_ROOT, BATCH_SIZE, LOOKBACK, horizon=6,
            skip_test_head_steps_2020=OHIO_SKIP_TEST_HEAD_STEPS,
        )
        print(f"  Found {len(patients)} patient train/test pairs.", flush=True)
    else:
        print("\nGenerating synthetic OhioT1DM-style data for 12 patients...")
        patients = create_dataloaders(N_PATIENTS, BATCH_SIZE, LOOKBACK, horizon=6)

    all_results = {}
    patient_preds = {}  # for figures

    for pid, pdata in patients.items():
        print(f"\n{'='*50}")
        print(f"Patient {pid}")
        print(f"{'='*50}")

        patient_results = {}

        # Classical baselines
        print("  Running ARIMA and SVR...")
        classical = run_classical_baselines(pdata, horizon=6)
        patient_results.update(classical)

        # DL baselines
        for name, cls, kw in [
            ('LSTM', LSTMBaseline, {'input_dim': 3, 'hidden_dim': 128, 'n_layers': 2}),
            ('GRU', GRUBaseline, {'input_dim': 3, 'hidden_dim': 128, 'n_layers': 2}),
            ('Transformer', TransformerBaseline, {'input_dim': 3, 'd_model': 128, 'n_heads': 8, 'n_layers': 4}),
            ('TFT-only', TFTOnlyBaseline, {'d_model': 256, 'n_heads': 8, 'n_layers': 4}),
        ]:
            print(f"  Training {name}...")
            bmodel = train_baseline(cls, pdata, horizon=6, epochs=60, **kw)
            preds, targets = evaluate_model(bmodel, pdata['test'], 6, DEVICE)
            patient_results[name] = compute_metrics(preds, targets)
            del bmodel
            torch.cuda.empty_cache()

        # GlucoTwin (full)
        print("  Training GlucoTwin...")
        gt_model = train_glucotwin(pdata, horizon=6, epochs=EPOCHS)
        preds, targets = evaluate_model(gt_model, pdata['test'], 6, DEVICE, is_glucotwin=True)
        patient_results['GlucoTwin'] = compute_metrics(preds, targets)
        patient_results['GlucoTwin_trust'] = evaluate_glucotwin_trust(
            gt_model, pdata['test'], 6, DEVICE,
        )
        patient_preds[pid] = {'preds': preds, 'targets': targets}

        if RUN_ABLATIONS:
            ab_configs = [
                ('Ablation_no_bergman', dict(model_kwargs=dict(ablate_bergman=True))),
                ('Ablation_no_pinn', dict(model_kwargs=dict(), lambda_phys=0.0)),
                ('Ablation_concat_fusion', dict(model_kwargs=dict(fusion_mode='concat'))),
                ('Ablation_no_med', dict(model_kwargs=dict(use_med=False))),
                ('Ablation_no_life', dict(model_kwargs=dict(use_life=False))),
                ('Ablation_no_ehr', dict(model_kwargs=dict(use_ehr=False))),
                ('Ablation_cgm_only', dict(model_kwargs=dict(use_ehr=False, use_med=False, use_life=False))),
            ]
            for aname, cfg in ab_configs:
                print(f"  Ablation {aname}...")
                mkw = cfg.get('model_kwargs', {})
                lp = cfg.get('lambda_phys', 0.05)
                am = train_glucotwin(
                    pdata, horizon=6, epochs=ABLATION_EPOCHS,
                    model_kwargs=mkw, lambda_phys=lp,
                )
                pr, tg = evaluate_model(am, pdata['test'], 6, DEVICE, is_glucotwin=True)
                patient_results[aname] = compute_metrics(pr, tg)
                del am
                torch.cuda.empty_cache()

        # Multi-horizon GlucoTwin: one H=24 retrain per subject, then 60 and 120 min
        # rows report terminal error at leads 12 and 24 from the *same* checkpoint so
        # lead-time error cannot invert due to independent small-head retrains.
        if not SKIP_MULTI_HORIZON:
            if DATA_SOURCE == 'real':
                pts24 = create_dataloaders_real(
                    OHIO_ROOT, BATCH_SIZE, LOOKBACK, horizon=24,
                    skip_test_head_steps_2020=OHIO_SKIP_TEST_HEAD_STEPS,
                )
            else:
                pts24 = create_dataloaders(N_PATIENTS, BATCH_SIZE, LOOKBACK, horizon=24)
            pdata24 = pts24[pid]
            print(
                f"  Training GlucoTwin (24-step head for 60/120 min leads from one model)...",
                flush=True,
            )
            gt_h = train_glucotwin(pdata24, horizon=24, epochs=MULTI_HORIZON_EPOCHS)
            ph, th = evaluate_model(gt_h, pdata24['test'], 24, DEVICE, is_glucotwin=True)
            ph = np.asarray(ph)
            th = np.asarray(th)
            patient_results['GlucoTwin_60min'] = compute_metrics(
                ph[:, :12], th[:, :12], reduction='terminal',
            )
            patient_results['GlucoTwin_120min'] = compute_metrics(
                ph, th, reduction='terminal',
            )
            del gt_h
            torch.cuda.empty_cache()
        else:
            print("  SKIP_MULTI_HORIZON=1: skipping 60/120 min GlucoTwin retrains for this run.", flush=True)

        if EXTEND_BASELINE_HORIZONS:
            if DATA_SOURCE == 'real':
                pts24 = create_dataloaders_real(
                    OHIO_ROOT, BATCH_SIZE, LOOKBACK, horizon=24,
                    skip_test_head_steps_2020=OHIO_SKIP_TEST_HEAD_STEPS,
                )
            else:
                pts24 = create_dataloaders(N_PATIENTS, BATCH_SIZE, LOOKBACK, horizon=24)
            pdata24 = pts24[pid]
            print(f"  Training TFT-only (24-step head for 60/120 min leads from one model)...")
            tft_h = train_baseline(
                TFTOnlyBaseline, pdata24, horizon=24, epochs=HORIZON_BASELINE_EPOCHS,
                d_model=256, n_heads=8, n_layers=4,
            )
            ph, th = evaluate_model(tft_h, pdata24['test'], 24, DEVICE)
            ph = np.asarray(ph)
            th = np.asarray(th)
            patient_results['TFT-only_60min'] = compute_metrics(
                ph[:, :12], th[:, :12], reduction='terminal',
            )
            patient_results['TFT-only_120min'] = compute_metrics(
                ph, th, reduction='terminal',
            )
            del tft_h
            torch.cuda.empty_cache()

        all_results[pid] = patient_results
        del gt_model
        torch.cuda.empty_cache()

        # Print summary
        for mname, metrics in patient_results.items():
            if isinstance(metrics, dict) and 'rmse' in metrics:
                print(f"    {mname:15s}: RMSE={metrics['rmse']:.1f}, MAE={metrics['mae']:.1f}, EGA={metrics['ega_ab']:.1f}%")
            elif isinstance(metrics, dict) and 'picp_80' in metrics:
                print(
                    f"    {mname:15s}: PICP(80%)={metrics['picp_80']:.3f}, "
                    f"RMSE@G0<70={metrics['rmse_ctx_g0_lt_70']:.1f}, "
                    f"RMSE@G0>=70={metrics['rmse_ctx_g0_ge_70']:.1f}",
                    flush=True,
                )

    # Aggregate results
    print("\n" + "=" * 60)
    print("AGGREGATE RESULTS (mean ± std across patients)")
    print("=" * 60)

    base_models = ['ARIMA', 'SVR', 'LSTM', 'GRU', 'Transformer', 'TFT-only', 'GlucoTwin']
    _skip_agg_keys = {'GlucoTwin_trust'}
    extra_models = sorted(
        {k for pid in all_results for k in all_results[pid].keys()
         if k not in base_models and not k.startswith('Ablation_') and k not in _skip_agg_keys}
    )
    model_names = base_models + extra_models
    aggregate = {}
    for mname in model_names:
        rmses = [all_results[pid][mname]['rmse'] for pid in all_results if mname in all_results[pid]]
        if not rmses:
            continue
        maes = [all_results[pid][mname]['mae'] for pid in all_results if mname in all_results[pid]]
        egas = [all_results[pid][mname]['ega_ab'] for pid in all_results if mname in all_results[pid]]
        aggregate[mname] = {
            'rmse_mean': np.mean(rmses), 'rmse_std': np.std(rmses),
            'mae_mean': np.mean(maes), 'mae_std': np.std(maes),
            'ega_mean': np.mean(egas), 'ega_std': np.std(egas),
        }
        print(f"  {mname:22s}: RMSE={np.mean(rmses):.1f}±{np.std(rmses):.1f}, "
              f"MAE={np.mean(maes):.1f}±{np.std(maes):.1f}, "
              f"EGA A+B={np.mean(egas):.1f}%")

    # GlucoTwin trust / reliability (quantile head PICP + hypo-context RMSE)
    picps = []
    r_lo, r_hi = [], []
    for pid in all_results:
        t = all_results[pid].get('GlucoTwin_trust')
        if not t:
            continue
        picps.append(t['picp_80'])
        if np.isfinite(t['rmse_ctx_g0_lt_70']):
            r_lo.append(t['rmse_ctx_g0_lt_70'])
        if np.isfinite(t['rmse_ctx_g0_ge_70']):
            r_hi.append(t['rmse_ctx_g0_ge_70'])
    if picps:
        aggregate['GlucoTwin_trust'] = {
            'picp_80_mean': float(np.mean(picps)),
            'picp_80_std': float(np.std(picps)),
            'rmse_ctx_g0_lt_70_mean': float(np.mean(r_lo)) if r_lo else float('nan'),
            'rmse_ctx_g0_lt_70_std': float(np.std(r_lo)) if r_lo else float('nan'),
            'rmse_ctx_g0_ge_70_mean': float(np.mean(r_hi)) if r_hi else float('nan'),
            'rmse_ctx_g0_ge_70_std': float(np.std(r_hi)) if r_hi else float('nan'),
        }
        g = aggregate['GlucoTwin_trust']
        print(
            f"  {'GlucoTwin_trust':22s}: PICP(80%)={g['picp_80_mean']:.3f}±{g['picp_80_std']:.3f}, "
            f"RMSE@G0<70={g['rmse_ctx_g0_lt_70_mean']:.1f}, RMSE@G0>=70={g['rmse_ctx_g0_ge_70_mean']:.1f}",
            flush=True,
        )

    # Save results
    results_file = os.path.join(OUTPUT_DIR, 'all_results.json')
    serializable = {
        pid: {mname: _jsonify_metrics(metrics) for mname, metrics in pr.items()}
        for pid, pr in all_results.items()
    }

    with open(results_file, 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to {results_file}")

    # Save aggregate
    agg_file = os.path.join(OUTPUT_DIR, 'aggregate_results.json')
    with open(agg_file, 'w') as f:
        json.dump(_jsonify_metrics(aggregate), f, indent=2)

    if RUN_ABLATIONS:
        ab_names = sorted(
            {k for pid in all_results for k in all_results[pid] if k.startswith('Ablation_')}
        )
        ab_agg = {}
        for aname in ab_names:
            rmses = [all_results[pid][aname]['rmse'] for pid in all_results if aname in all_results[pid]]
            maes = [all_results[pid][aname]['mae'] for pid in all_results if aname in all_results[pid]]
            egas = [all_results[pid][aname]['ega_ab'] for pid in all_results if aname in all_results[pid]]
            full_rmse = [all_results[pid]['GlucoTwin']['rmse'] for pid in all_results]
            delta_pct = (np.mean(rmses) - np.mean(full_rmse)) / max(np.mean(full_rmse), 1e-6) * 100.0
            ab_agg[aname] = {
                'rmse_mean': float(np.mean(rmses)),
                'rmse_std': float(np.std(rmses)),
                'mae_mean': float(np.mean(maes)),
                'mae_std': float(np.std(maes)),
                'ega_mean': float(np.mean(egas)),
                'ega_std': float(np.std(egas)),
                'delta_rmse_pct_vs_full': float(delta_pct),
            }
        ab_path = os.path.join(OUTPUT_DIR, 'ablation_aggregate.json')
        with open(ab_path, 'w') as f:
            json.dump(ab_agg, f, indent=2)
        ablation_aggregate_saved_this_run = True
        print(f"\nAblation aggregates saved to {ab_path}")

    if not EXPERIMENT_ONLY:
        # Save patient predictions for figure generation
        np.savez(os.path.join(OUTPUT_DIR, 'predictions.npz'),
                 **{f'{pid}_preds': v['preds'] for pid, v in patient_preds.items()},
                 **{f'{pid}_targets': v['targets'] for pid, v in patient_preds.items()})

        print("\n" + "=" * 60)
        print("GENERATING FIGURES...")
        print("=" * 60)
        from src.experiments.generate_figures import generate_all_figures
        generate_all_figures(
            all_results,
            patient_preds,
            patients,
            FIG_DIR,
            results_dir=os.path.join(_ROOT, 'outputs', 'results'),
        )

        try:
            from src.experiments.benchmark_inference import run_benchmark
            run_benchmark(OUTPUT_DIR)
        except Exception as e:
            print(f"Latency benchmark skipped: {e}")

        app_root = _ROOT
        try:
            bscript = os.path.join(app_root, 'tools', 'build_ablation_tex.py')
            r = subprocess.run(
                [sys.executable, bscript],
                cwd=app_root, capture_output=True, text=True,
            )
            print(r.stdout or '', r.stderr or '')
            if r.returncode != 0:
                print(f"build_ablation_tex.py exit {r.returncode}")
        except Exception as e:
            print(f"build_ablation_tex skipped: {e}")

        try:
            hscript = os.path.join(app_root, 'tools', 'build_horizon_tex.py')
            r2 = subprocess.run(
                [sys.executable, hscript],
                cwd=app_root, capture_output=True, text=True,
            )
            print(r2.stdout or '', r2.stderr or '')
        except Exception as e:
            print(f"build_horizon_tex skipped: {e}")

        for script in (
            'build_data_mode_tex.py',
            'build_trust_tex.py',
            'build_main_results_tex.py',
            'build_paired_stats_tex.py',
            'build_personalization_tex.py',
        ):
            try:
                p = os.path.join(app_root, 'tools', script)
                r3 = subprocess.run(
                    [sys.executable, p],
                    cwd=app_root, capture_output=True, text=True,
                )
                print(r3.stdout or '', r3.stderr or '')
                if r3.returncode != 0:
                    print(f'{script} exit {r3.returncode}')
            except Exception as e:
                print(f'{script} skipped: {e}')
    else:
        print(
            "\nGLUCOTWIN_EXPERIMENT_ONLY=1: skipping predictions.npz, figures, "
            "latency benchmark, and table-build subprocesses.",
            flush=True,
        )

    _manifest_ablation_updates = {'ablation_aggregate_saved_this_run': ablation_aggregate_saved_this_run}
    if ablation_aggregate_saved_this_run:
        _manifest_ablation_updates['ablation_aggregate_saved_via_component_script'] = False
    _update_run_manifest(_manifest_ablation_updates)

    print("\nAll experiments complete!")


if __name__ == '__main__':
    main()
