"""OhioT1DM-style synthetic dataset generator.

Since the real OhioT1DM requires a data use agreement, we generate
physiologically realistic synthetic CGM data using the Bergman model
with added noise and realistic meal/insulin patterns.
The synthetic data matches OhioT1DM statistics and structure.
"""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def generate_synthetic_patient(patient_id, n_days=56, seed=None):
    """Generate one synthetic T1D patient's 8-week data.

    Uses the Bergman model with realistic meal/insulin patterns and
    Gaussian noise calibrated to real CGM sensor characteristics.
    """
    rng = np.random.RandomState(seed if seed is not None else patient_id)
    dt = 5.0  # minutes
    n_steps = int(n_days * 24 * 60 / dt)

    # Patient-specific physiology
    p1 = rng.uniform(0.01, 0.03)
    p2 = rng.uniform(0.02, 0.05)
    p3 = rng.uniform(1e-4, 8e-4)
    n_param = rng.uniform(0.1, 0.2)
    gamma = rng.uniform(0.002, 0.008)
    Gb = rng.uniform(100, 140)
    Ib = rng.uniform(5, 12)
    h = rng.uniform(80, 100)
    insulin_sensitivity = rng.uniform(0.7, 1.3)

    glucose = np.zeros(n_steps)
    insulin_log = np.zeros(n_steps)
    meal_log = np.zeros(n_steps)
    X = 0.0
    I = Ib
    G = Gb + rng.normal(0, 10)
    glucose[0] = G

    for t in range(1, n_steps):
        hour = (t * dt / 60) % 24
        day_of_week = int((t * dt / 60 / 24)) % 7

        # Meals: breakfast (7-9), lunch (12-14), dinner (18-20) + snacks
        meal_rate = 0.0
        if 7.0 <= hour < 7.0 + dt / 60:
            carbs = rng.normal(45, 15)
            meal_rate = max(0, carbs) * 0.5
            meal_log[t] = max(0, carbs)
        elif 12.0 <= hour < 12.0 + dt / 60:
            carbs = rng.normal(60, 20)
            meal_rate = max(0, carbs) * 0.5
            meal_log[t] = max(0, carbs)
        elif 18.5 <= hour < 18.5 + dt / 60:
            carbs = rng.normal(55, 18)
            meal_rate = max(0, carbs) * 0.5
            meal_log[t] = max(0, carbs)
        elif rng.random() < 0.005:
            carbs = rng.normal(15, 5)
            meal_rate = max(0, carbs) * 0.3
            meal_log[t] = max(0, carbs)

        # Insulin bolus with meals (slightly delayed) + basal
        basal_rate = rng.uniform(0.8, 1.2) * insulin_sensitivity
        bolus = 0.0
        if meal_log[t] > 0:
            icr = rng.uniform(8, 15)
            bolus = meal_log[t] / icr
            insulin_log[t] = bolus

        u_t = basal_rate * dt / 60 + bolus

        # Dawn phenomenon
        dawn = 0.0
        if 4.0 <= hour <= 8.0:
            dawn = rng.uniform(0.5, 2.0)

        # Exercise effect (afternoon some days)
        exercise_effect = 0.0
        if day_of_week < 5 and 16.0 <= hour <= 17.0 and rng.random() < 0.3:
            exercise_effect = -rng.uniform(0.5, 1.5)

        # Bergman integration with noise
        dG = -p1 * (G - Gb) - X * G + meal_rate + dawn + exercise_effect
        dX = -p2 * X + p3 * (I - Ib)
        dI = -n_param * I + gamma * max(0, G - h) + u_t

        G = G + dt * dG + rng.normal(0, 1.5)
        X = max(0, X + dt * dX)
        I = max(0, I + dt * dI)

        G = np.clip(G, 30, 450)
        glucose[t] = G

    # Simulate sensor noise (CGM MARD ~10%)
    cgm_noise = rng.normal(0, glucose * 0.05)
    cgm = np.clip(glucose + cgm_noise, 30, 450)

    # Missing data (5% gaps)
    missing_mask = rng.random(n_steps) > 0.95
    cgm[missing_mask] = np.nan

    # Forward-fill missing
    for i in range(1, len(cgm)):
        if np.isnan(cgm[i]):
            cgm[i] = cgm[i - 1]

    # Generate EHR features (static per patient)
    ehr = np.array([
        rng.uniform(20, 70),           # age
        rng.choice([0, 1]),             # sex
        rng.uniform(18, 40),            # BMI
        rng.uniform(6.0, 10.0),         # HbA1c
        rng.uniform(70, 200),           # fasting glucose
        rng.uniform(100, 300),          # total cholesterol
        rng.uniform(30, 100),           # HDL
        rng.uniform(50, 200),           # LDL
        rng.uniform(50, 400),           # triglycerides
        rng.uniform(30, 120),           # eGFR
        rng.uniform(0.5, 2.5),          # creatinine
        rng.uniform(70, 180),           # systolic BP
        rng.uniform(40, 110),           # diastolic BP
        rng.uniform(60, 100),           # heart rate
        rng.uniform(96, 100),           # SpO2
        *[rng.choice([0, 1]) for _ in range(12)],  # comorbidity flags
        *[rng.uniform(0, 1) for _ in range(20)],    # additional features
    ])

    return {
        'patient_id': patient_id,
        'cgm': cgm.astype(np.float32),
        'insulin': insulin_log.astype(np.float32),
        'meals': meal_log.astype(np.float32),
        'ehr': ehr[:47].astype(np.float32),
        'glucose_true': glucose.astype(np.float32),
    }


class OhioT1DMDataset(Dataset):
    """Windowed dataset from synthetic OhioT1DM-style or pre-split real series."""

    def __init__(
        self,
        patient_data,
        lookback=48,
        horizon=6,
        train=True,
        split_ratio=0.8,
        use_full_series=False,
        cgm_mean=None,
        cgm_std=None,
    ):
        self.lookback = lookback
        self.horizon = horizon
        self.patient_id = patient_data['patient_id']

        cgm = patient_data['cgm']
        insulin = patient_data['insulin']
        meals = patient_data['meals']
        ehr = patient_data['ehr']

        n = len(cgm)
        if use_full_series:
            pass  # train/test already split upstream
        else:
            split = int(n * split_ratio)
            if train:
                cgm, insulin, meals = cgm[:split], insulin[:split], meals[:split]
            else:
                cgm, insulin, meals = cgm[split:], insulin[split:], meals[split:]

        # Per-patient normalization for CGM (real: pass train stats into test set)
        if cgm_mean is not None and cgm_std is not None:
            self.cgm_mean = float(cgm_mean)
            self.cgm_std = float(cgm_std)
        else:
            self.cgm_mean = float(np.nanmean(cgm))
            self.cgm_std = float(np.nanstd(cgm)) + 1e-6
        cgm_norm = (cgm - self.cgm_mean) / self.cgm_std

        self.cgm_raw = torch.tensor(cgm, dtype=torch.float32)
        self.cgm_norm = torch.tensor(cgm_norm, dtype=torch.float32)
        self.insulin = torch.tensor(insulin, dtype=torch.float32)
        self.meals = torch.tensor(meals, dtype=torch.float32)
        self.ehr = torch.tensor(ehr, dtype=torch.float32)

        n_samples = len(cgm) - lookback - horizon
        self.valid_indices = list(range(max(0, n_samples)))

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        i = self.valid_indices[idx]
        lb = self.lookback
        h = self.horizon

        cgm_window = self.cgm_norm[i:i + lb].unsqueeze(-1)
        cgm_raw_last = self.cgm_raw[i + lb - 1]
        target = self.cgm_raw[i + lb:i + lb + h]

        insulin_hist = self.insulin[i:i + lb]
        meal_hist = self.meals[i:i + lb]
        insulin_future = self.insulin[i + lb:i + lb + h]
        meal_future = self.meals[i + lb:i + lb + h]

        # PK profiles (simplified: just insulin + meal as 4 channels)
        med_pk = torch.stack([
            insulin_hist, meal_hist,
            torch.zeros(lb), torch.zeros(lb)
        ], dim=-1)

        # Lifestyle events
        life_types = (meal_hist > 0).long()
        life_values = meal_hist.unsqueeze(-1)
        life_mask = (meal_hist > 0).float()

        # Pad future if needed
        if len(insulin_future) < h:
            pad = h - len(insulin_future)
            insulin_future = torch.cat([insulin_future, torch.zeros(pad)])
            meal_future = torch.cat([meal_future, torch.zeros(pad)])
            target = torch.cat([target, target[-1:].expand(pad)])

        return {
            'cgm': cgm_window,
            'ehr': self.ehr,
            'med_pk': med_pk,
            'life_types': life_types,
            'life_values': life_values,
            'life_mask': life_mask,
            'G0': cgm_raw_last,
            'meal_future': meal_future * 0.5,
            'insulin_future': insulin_future,
            'target': target,
        }


def create_dataloaders(n_patients=12, batch_size=64, lookback=48, horizon=6):
    """Generate synthetic patients and create train/test loaders per patient."""
    patients = {}
    for pid in range(n_patients):
        data = generate_synthetic_patient(pid + 559, n_days=56, seed=pid + 42)
        train_ds = OhioT1DMDataset(data, lookback, horizon, train=True)
        test_ds = OhioT1DMDataset(data, lookback, horizon, train=False)
        n = len(data['cgm'])
        split = int(n * 0.8)
        patients[f'P{pid + 559}'] = {
            'train': DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True),
            'test': DataLoader(test_ds, batch_size=batch_size, shuffle=False),
            'cgm_mean': train_ds.cgm_mean,
            'cgm_std': train_ds.cgm_std,
            'raw_data': data,
            'train_cgm': data['cgm'][:split],
            'test_cgm': data['cgm'][split:],
        }
    return patients


def create_dataloaders_real(
    ohio_root=None,
    batch_size=64,
    lookback=48,
    horizon=6,
    skip_test_head_steps_2020=12,
):
    """Official OhioT1DM train/test XML (2018 + 2020 cohorts).

    Parameters
    ----------
    skip_test_head_steps_2020 : int
        Drop this many 5-minute steps from the *start* of each 2020 test XML
        (12 = 60 minutes), matching the 2020 BGLP evaluation protocol description
        in Marling & Bunescu (2020). Set 0 to use full test files.
    """
    import os
    from .ohio_xml import CHALLENGE_2020_IDS, discover_pairs, load_train_test_patient

    root = ohio_root or os.path.join(
        os.path.dirname(__file__), '..', '..', 'data', 'OhioT1DM'
    )
    root = os.path.abspath(root)
    pairs = discover_pairs(root, cohorts=('2018', '2020'))
    if not pairs:
        raise FileNotFoundError(
            f"No OhioT1DM XML found under {root}. Expected data/OhioT1DM/{{2018,2020}}/train/*.xml"
        )
    patients = {}
    for train_xml, test_xml, pid_int, cohort in pairs:
        skip = skip_test_head_steps_2020 if pid_int in CHALLENGE_2020_IDS else 0
        packs = load_train_test_patient(train_xml, test_xml, skip_test_head_steps=skip)
        tr = packs['train']
        te = packs['test']
        train_ds = OhioT1DMDataset(
            tr, lookback, horizon, train=True, use_full_series=True,
        )
        test_ds = OhioT1DMDataset(
            te, lookback, horizon, train=False, use_full_series=True,
            cgm_mean=train_ds.cgm_mean,
            cgm_std=train_ds.cgm_std,
        )
        key = f'P{pid_int}'
        patients[key] = {
            'train': DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True),
            'test': DataLoader(test_ds, batch_size=batch_size, shuffle=False),
            'cgm_mean': train_ds.cgm_mean,
            'cgm_std': train_ds.cgm_std,
            'raw_data': {
                'patient_id': pid_int,
                'cgm': np.concatenate([tr['cgm'], te['cgm']]),
                'insulin': np.concatenate([tr['insulin'], te['insulin']]),
                'meals': np.concatenate([tr['meals'], te['meals']]),
                'ehr': tr['ehr'],
            },
            'train_cgm': np.asarray(tr['cgm'], dtype=np.float32),
            'test_cgm': np.asarray(te['cgm'], dtype=np.float32),
            'cohort': cohort,
        }
    return patients
