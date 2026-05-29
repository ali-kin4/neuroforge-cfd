# NeuroForge CFD — Build Conventions & Interface Contract

**Every agent must read this file, plus `src/neuroforge/core/types.py`,
`src/neuroforge/core/config.py`, and `src/neuroforge/models/base.py`, before
writing code.** These four files are the *frozen contract*. Do not edit them.
Build only the files assigned to you. Import other modules strictly through the
public signatures listed here.

---

## 0. Global rules

- **Python ≥ 3.10**, `from __future__ import annotations` at the top of every module.
- **Device-agnostic, CPU-first.** No hard-coded `.cuda()`. Use the tensor's own
  device/dtype. Default dtype is `float32`. Everything must run on CPU.
- **No heavy work at import time.** Importing a module must not train, download,
  or allocate large arrays.
- Type hints on all public functions. NumPy-style docstrings. Keep it readable.
- Arrays: structured fields are `(ny, nx)` numpy `float32`. Network tensors are
  channel-first `(B, C, ny, nx)` torch tensors.
- `INPUT_CHANNELS` (7): `(sdf, mask, x, y, u_in, v_in, log_re)`.
  `OUTPUT_CHANNELS` (4): `(u, v, p, nut)`. `N_IN = 7`, `N_OUT = 4`.
- Pressure `p` is **kinematic** (p/ρ). Physics residuals operate on **physical
  (denormalised)** fields. `nu_eff = nu_laminar + nut`.
- Optional deps (`airfrans`, `pyvista`, `streamlit`, `plotly`) must be imported
  lazily inside the function that needs them, with a helpful error if missing.
- Tests run on CPU in seconds. Keep default model sizes tiny.

## Coordinate / grid convention

```python
domain = Domain(bounds=(xmin, xmax, ymin, ymax), nx=128, ny=128)
X, Y = domain.grid()          # (ny, nx) via meshgrid(..., indexing="xy")
dx, dy = domain.dx, domain.dy
```

---

## 1. `geometry/` and `data/`  — owned by **Agent G-DATA**

### `geometry/airfoil.py`
```python
def naca_airfoil(code: str = "naca2412", n_points: int = 200,
                 closed: bool = True) -> Geometry:
    """NACA 4- and 5-digit airfoils. Chord = 1, leading edge at x=0, TE at x=1.
    Returns Geometry with ordered CCW surface_points (N,2) and surface_normals."""

def airfoil_from_dat(path: str, name: str | None = None) -> Geometry:
    """Load a Selig/Lednicer .dat coordinate file into a Geometry."""
```

### `geometry/sdf.py`
```python
def signed_distance(geom: Geometry, domain: Domain) -> np.ndarray:
    """(ny,nx) signed distance to the surface; NEGATIVE inside the solid body."""

def solid_mask(geom: Geometry, domain: Domain) -> np.ndarray:
    """(ny,nx) float32: 1.0 in fluid, 0.0 inside the body."""

def surface_normals(geom: Geometry) -> np.ndarray:
    """(N,2) outward unit normals for the geometry's surface points."""
```

### `geometry/encode.py`  — **critical: produces network input**
```python
def encode_case(case: FlowCase) -> np.ndarray:
    """Build the (N_IN, ny, nx) input stack in INPUT_CHANNELS order:
    [sdf, mask, x_norm, y_norm, u_in_field, v_in_field, log10_re].
    x_norm/y_norm are coordinates scaled to ~[-1,1]. u_in/v_in come from
    case.bc.inlet_vector() broadcast over the grid. log_re = log10(bc.reynolds)."""

def case_geometry_fields(case: FlowCase) -> tuple[np.ndarray, np.ndarray]:
    """Convenience: returns (sdf, mask) (ny,nx) for a case."""
```

### `geometry/io.py`
```python
def load_stl(path: str) -> Geometry          # 2D slice / projection; lazy-import optional libs
def load_obj(path: str) -> Geometry
```
STL/OBJ may raise `NotImplementedError("planned for v0.2")` if a dependency is
absent — but must be importable.

### `data/synthetic.py`  — **must run with zero downloads**
```python
class SyntheticRANS:
    """Generates physically-plausible 2D flow fields around airfoils on the grid
    WITHOUT a real solver — superposition of uniform flow + doublet/vortex
    (potential flow) plus a viscous wake & boundary-layer model, so continuity is
    approximately satisfied and fields look RANS-like. CPU-cheap."""
    def __init__(self, resolution: int = 128, seed: int = 0,
                 domain_bounds=(-1.0, 2.0, -1.5, 1.5)): ...
    def sample_case(self, idx: int) -> FlowCase: ...
    def solve(self, case: FlowCase) -> FlowField:
        """Analytic pseudo-RANS field consistent with the case (u,v,p,nut,mask,sdf)."""
    def generate(self, n: int) -> list[tuple[FlowCase, FlowField]]: ...
```

### `data/airfrans_loader.py`
```python
def download_airfrans(root: str = "data") -> str: ...   # lazy import airfrans
def load_airfrans(root: str = "data", task: str = "scarce",
                  train: bool = True) -> list[tuple[FlowCase, FlowField]]:
    """Load AirfRANS sims, rasterise point clouds onto the structured grid, and
    return (FlowCase, FlowField) pairs. Lazy-imports `airfrans`."""
```

### `data/rasterize.py`
```python
def rasterize_point_cloud(points: np.ndarray, values: np.ndarray,
                          domain: Domain, fill: float = 0.0,
                          method: str = "linear") -> np.ndarray:
    """Scatter (M,2) points with (M,K) values onto a (K,ny,nx) grid (scipy.griddata)."""
```

### `data/datamodule.py`  — **bridges to training**
```python
class Normalizer:
    """Per-channel mean/std standardisation. Fit on training fields.
    Stores stats for input (N_IN) and output (N_OUT) channels separately."""
    def fit(self, inputs: np.ndarray, outputs: np.ndarray) -> "Normalizer": ...
    def norm_in(self, x): ...        # accepts/returns np or torch, same shape
    def norm_out(self, y): ...
    def denorm_out(self, y): ...
    def state_dict(self) -> dict: ...
    @classmethod
    def from_state_dict(cls, d: dict) -> "Normalizer": ...

class FlowDataset(torch.utils.data.Dataset):
    """Wraps a list[(FlowCase, FlowField)]. __getitem__ returns dict with keys:
    'input' (N_IN,ny,nx) float32 tensor, 'target' (N_OUT,ny,nx),
    'mask' (1,ny,nx), 'sdf' (1,ny,nx), and 'case' (the FlowCase, via collate
    that keeps it as a python list). Uses encode_case for inputs."""
    def __init__(self, pairs, normalizer: "Normalizer | None" = None): ...

def build_dataloaders(cfg: DataConfig) -> tuple[DataLoader, DataLoader, Normalizer]:
    """Create train/val loaders + a fitted Normalizer from a DataConfig
    (source='synthetic' or 'airfrans'). Default collate keeps 'case' as a list."""
```

---

## 2. `models/`  — owned by **Agent MODELS** (do NOT edit `models/base.py`)

All backbones subclass `NeuralSolver` (from `neuroforge.models.base`), accept
`in_channels=N_IN, out_channels=N_OUT, **hp` and register via
`@register_model("name")`. `forward(x:(B,N_IN,H,W)) -> (B,N_OUT,H,W)`.

Required registry keys & classes:
- `@register_model("fno")`   → `class FNO2d(NeuralSolver)` (Fourier Neural Operator; ctor:
  `width=32, n_layers=4, modes=16, dropout=0.0`).
- `@register_model("geo_fno")` → `class GeoFNO(NeuralSolver)` (learned-deformation Geo-FNO;
  may wrap FNO2d; falls back gracefully).
- `@register_model("transformer")` → `class PhysicsTransformer(NeuralSolver)`
  (Transolver-style **physics-attention** with learnable slices/tokens, linear
  attention over `n_slices`; ctor: `width=32, n_layers=4, n_heads=4, n_slices=32`).
- `@register_model("unet")`  → `class UNet(NeuralSolver)` (baseline).
- `@register_model("deeponet")` → `class DeepONet(NeuralSolver)` (grid-to-grid DeepONet
  baseline; branch sees input stack, trunk sees coordinates).

Correction & UQ:
- `class LocalCorrectionNet(CorrectionNetwork)` — small CNN, ctor `width=24,
  n_layers=3`. `forward(field, residual, geom) -> delta (B,N_OUT,H,W)`. Input is
  the concat of `[field (N_OUT), residual (3), geom (N_IN)]` → `N_OUT` out.
- `class DeepEnsemble` — wraps a list of `NeuralSolver`; `predict_with_uncertainty(x)
  -> (mean (B,N_OUT,H,W), std (B,N_OUT,H,W))`.
- `class MCDropoutUQ` — wraps one dropout-enabled `NeuralSolver`;
  `predict_with_uncertainty(x, n_samples=16) -> (mean, std)` (keeps dropout on).

`models/__init__.py` already imports these names inside a `try/except` — keep the
class names exactly as above.

---

## 3. `physics/`  — owned by **Agent PHYSICS**

### `physics/operators.py`  — backend-agnostic (numpy **or** torch via duck typing)
```python
def ddx(f, dx): ...        # d/dx, central diff, one-sided at edges; same shape as f
def ddy(f, dy): ...
def laplacian(f, dx, dy): ...
def divergence(u, v, dx, dy): ...
def gradient(f, dx, dy) -> tuple: ...   # (df/dx, df/dy)
```
Accept arrays shaped `(..., ny, nx)`. Must work for both numpy ndarray and torch
Tensor (use slicing; pick `np`/`torch` by `type(f)`).

### `physics/residuals.py`  — **the verifier**
```python
def continuity_residual(field: FlowField) -> np.ndarray: ...       # du/dx+dv/dy
def momentum_residual(field: FlowField, fluid: FluidProperties) -> tuple[np.ndarray, np.ndarray]:
    """(r_x, r_y) steady incompressible RANS momentum residuals using nu_eff=nu+nut."""
def bc_violation(field: FlowField, case: FlowCase) -> np.ndarray:
    """No-slip on the body (velocity should ->0 at mask boundary) + far-field inlet mismatch."""

class PhysicsChecker:
    """Turns a FlowField + FlowCase (+ optional uncertainty map) into Diagnostics."""
    def __init__(self, cfg: PhysicsConfig | None = None): ...
    def residuals(self, field: FlowField, case: FlowCase) -> dict[str, np.ndarray]:
        """{'continuity','momentum_x','momentum_y','bc'} (ny,nx) on physical fields."""
    def diagnose(self, field: FlowField, case: FlowCase,
                 uncertainty: np.ndarray | None = None) -> Diagnostics:
        """Full Diagnostics incl. trust map. If uncertainty is None, use 0s."""

def physics_residual_torch(pred: torch.Tensor, inp: torch.Tensor,
                           dx: float, dy: float, nu: torch.Tensor | float,
                           ) -> dict[str, torch.Tensor]:
    """Differentiable residuals on NORMALISED-then-denormalised tensors for the
    training loss. pred=(B,N_OUT,H,W), inp=(B,N_IN,H,W). Returns
    {'continuity','momentum_x','momentum_y'} each (B,1,H,W)."""
```

### `physics/metrics.py`
```python
def pressure_coefficient(field: FlowField, case: FlowCase) -> np.ndarray: ...   # Cp on grid
def force_coefficients(field: FlowField, case: FlowCase) -> dict[str, float]:
    """Integrate surface pressure + wall shear -> {'cl','cd','cm'} (lift/drag/moment)."""
def field_errors(pred: FlowField, ref: FlowField) -> dict[str, float]:
    """Relative L2 errors per field {'u','v','p','speed'} masked to fluid."""

def wall_shear_stress(field: FlowField, case: FlowCase) -> np.ndarray: ...
```

### `physics/trust.py`
```python
def trust_map(residual_mag: np.ndarray, uncertainty: np.ndarray,
              cfg: PhysicsConfig) -> tuple[np.ndarray, np.ndarray]:
    """Returns (trust [0..1], trust_class {0,1,2}). Combine normalised residual &
    uncertainty using cfg weights/thresholds. 2=green,1=yellow,0=red."""
```

---

## 4. `solver/` and `train/`  — owned by **Agent ENGINE**

### `solver/engine.py`  — **the product surface + the breakthrough loop**
```python
class Predictor:
    """Bridges a trained torch model to the numpy FlowField world.
    Wraps (model, normalizer). predict(case) -> FlowField."""
    def __init__(self, model: NeuralSolver, normalizer, device: str = "cpu"): ...
    def predict(self, case: FlowCase) -> FlowField: ...        # uses geometry.encode_case
    def predict_tensor(self, x: torch.Tensor) -> torch.Tensor: ...

class NeuroForgeEngine:
    """The self-correcting engine. Composes a Predictor, a PhysicsChecker, an
    optional CorrectionNetwork (Neural Residual Iteration) and an optional
    uncertainty estimator + classical fallback."""
    def __init__(self, predictor: Predictor, checker: "PhysicsChecker",
                 corrector=None, uq=None, config: Config | None = None): ...
    @classmethod
    def from_checkpoint(cls, path: str, config: Config | None = None) -> "NeuroForgeEngine": ...
    @classmethod
    def pretrained(cls) -> "NeuroForgeEngine":
        """Load a bundled demo checkpoint if present, else train a tiny model on
        synthetic data and cache it under checkpoints/demo.pt."""
    def solve(self, case: FlowCase, max_iters: int | None = None) -> SolveResult:
        """Predict -> diagnose -> (Neural Residual Iteration: correct low-trust
        regions, re-diagnose, repeat until residual_tol or max_iters) -> optional
        uncertainty-gated classical patch -> compute metrics. Records per-iter
        history of {'residual_norm','max_uncertainty','trust_mean'}."""

def demo() -> SolveResult:
    """One-call end-to-end demo: build a synthetic-trained engine, solve a NACA
    case, return the result. Used by the CLI `demo` command and the e2e test."""
```

### `solver/correction_loop.py`
```python
def neural_residual_iteration(field, case, checker, corrector, cfg: CorrectionConfig,
                              encode_fn, uq=None) -> tuple[FlowField, list[dict]]:
    """Run the trust-gated correction loop. Returns (final_field, history)."""
```

### `solver/fallback.py`
```python
class ClassicalFallback:
    """Optional local classical-CFD patch. The 'stub' backend returns the region
    unchanged but reports what WOULD run; 'openfoam'/'su2' raise NotImplementedError
    with guidance. Must be importable with none installed."""
    def __init__(self, backend: str = "stub"): ...
    def patch(self, field: FlowField, case: FlowCase, region_mask: np.ndarray) -> FlowField: ...
```

### `train/losses.py`
```python
class CompositeLoss:
    """data MSE (masked) + physics_weight * residual loss + bc_weight * bc loss.
    Uses physics.physics_residual_torch. ctor takes TrainConfig + normalizer + nu."""
    def __call__(self, pred, target, inp, mask) -> tuple[torch.Tensor, dict[str, float]]: ...
```

### `train/trainer.py`
```python
class Trainer:
    def __init__(self, model: NeuralSolver, cfg: Config, normalizer, nu: float = 1.5e-5): ...
    def fit(self, train_loader, val_loader) -> dict:  # returns history
    def save(self, path: str) -> None:    # saves model state, normalizer, ModelConfig
    @staticmethod
    def load(path: str, map_location="cpu") -> tuple[NeuralSolver, "Normalizer", ModelConfig]: ...
```
**Checkpoint format** (a dict saved by `torch.save`): keys
`{"model_state": ..., "model_config": <ModelConfig asdict>, "normalizer": <state_dict>,
"nu": float, "neuroforge_version": str}`. `from_checkpoint` and `Trainer.load`
must agree on this.

---

## 5. `viz/`, `cli.py`, `app/`, `examples/`  — owned by **Agent UX**

### `viz/plots.py`
```python
def plot_field(field, key="speed", ax=None, **kw): ...          # returns matplotlib Axes
def plot_residual(diag, key="continuity", ax=None, **kw): ...
def plot_trust(diag, ax=None, **kw): ...                        # green/yellow/red heatmap
def plot_uncertainty(diag, ax=None, **kw): ...
def plot_cp(field, case, ax=None, ref=None, **kw): ...          # Cp vs x/c curve
def plot_convergence(history, ax=None, **kw): ...               # residual vs iteration
def overview_figure(result) -> "matplotlib.figure.Figure": ...  # multi-panel summary
```

### `viz/report.py`
```python
def build_report(result: SolveResult, path: str, title: str | None = None) -> str:
    """Write a self-contained HTML report (PNGs embedded as base64) + sibling PNGs.
    Returns the path written. This is what SolveResult.save_report calls."""
```

### `cli.py`  — `def main(argv=None) -> int`. Subcommands (argparse):
- `neuroforge demo` — run `engine.demo()`, print metrics, write a report.
- `neuroforge train --config cfg.yaml` — train + save checkpoint.
- `neuroforge predict --ckpt p.pt --airfoil naca2412 --aoa 5 --re 3e6 [--report out.html]`.
- `neuroforge benchmark` — call `benchmarks.run_benchmarks`.
- `neuroforge info` — print version, available models, device.

### `app/streamlit_app.py`
A Streamlit UI: pick/upload airfoil, set AoA/Re/U, run engine, show field +
residual + trust + Cp + convergence. Lazy-import streamlit; guard if missing.

### `examples/`
`quickstart.py`, `demo_synthetic.py` (train tiny + solve + report),
`train_airfoil.py` (AirfRANS path with instructions).

---

## 6. `tests/`, `benchmarks/`, `docs/`  — owned by **Agent QA-DOCS**

- `tests/` pytest, CPU, fast (tiny grids e.g. 32×32, 1–2 epochs). Cover:
  geometry/SDF, encode channel order, physics operators (analytic checks:
  divergence of uniform flow ≈ 0; laplacian of linear field ≈ 0), residuals,
  metrics, each model forward-shape, correction net, Normalizer round-trip,
  trainer 1-epoch, engine.solve end-to-end (assert residual_norm does NOT
  increase over iterations), report generation to a temp dir, CLI `info`/`demo`.
- `benchmarks/run_benchmarks.py`: `def run_benchmarks(cfg=None) -> dict` comparing
  fno/unet/deeponet/transformer on synthetic data: field errors, inference time,
  param count, residual norms. Prints a table; returns a dict.
- `docs/paper/neuroforge_cfd.md`: research paper draft (abstract, intro,
  related work [DoMINO, Transolver/++, Geo-FNO, AirfRANS, residual correctors,
  UQ], method [Neural Residual Iteration — the novelty], experiments plan,
  limitations, references). `docs/architecture.md`, `docs/ROADMAP.md` (3 stages).

---

## Integration order (the lead integrates)

1. core (done) → 2. geometry/data, physics, models (independent) →
3. solver/train (uses 1,2's interfaces) → 4. viz/cli/app, tests/docs.

If you discover a genuine contract gap, **leave a `# CONTRACT-NOTE:` comment**
at the call site describing what you needed; do not silently change a frozen
file. The lead will reconcile.
