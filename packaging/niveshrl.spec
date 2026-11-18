# PyInstaller spec for the NiveshRL desktop app (one-folder build).
#
#   .venv\Scripts\pyinstaller packaging\niveshrl.spec --noconfirm --distpath build\dist --workpath build\work
#
# Output: build\dist\NiveshRL\NiveshRL.exe plus its _internal\ folder. The research data
# (price panel, model predictions, RL checkpoints, configs) is NOT bundled into the exe;
# packaging\build.ps1 stages it as build\dist\NiveshRL\seed\, which the app copies to
# %LOCALAPPDATA%\NiveshRL on first run (Program Files is read-only).
#
# No torch / transformers since 18 Nov 2026: the app's models are LightGBM, logistic, HAR, PCA + k-means and rules
# (tests/test_no_torch.py runs the app with both blocked). Research-only modules that need torch are left out.
import os
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

ROOT = os.path.abspath(os.path.join(SPECPATH, ".."))

RESEARCH_ONLY = ("niveshrl.dashboard", "niveshrl.plots", "niveshrl.agents", "niveshrl.algos", "niveshrl.models",
                 "niveshrl.env", "niveshrl.rewards", "niveshrl.explain", "niveshrl.baselines",
                 "niveshrl.research.rankers", "niveshrl.research.sentiment", "niveshrl.research.swing_meta",
                 "niveshrl.intraday.dl.tcn", "niveshrl.intraday.engine_v2")
hidden = ([m for m in collect_submodules("niveshrl") if not m.startswith(RESEARCH_ONLY)]
          + ["niveshrl_core", "lightgbm", "sklearn.isotonic", "sklearn.linear_model", "pyarrow", "yfinance",
             "websockets", "PySide6.QtNetwork"])
datas = (collect_data_files("lightgbm")
         + collect_data_files("yfinance"))

a = Analysis(
    [os.path.join(SPECPATH, "launcher.py")],
    pathex=[os.path.join(ROOT, "src")],
    binaries=[],
    datas=datas,
    hiddenimports=hidden,
    excludes=["streamlit", "tensorflow", "jax", "flax", "keras", "plotly", "matplotlib", "tkinter", "IPython", "jupyter", "notebook", "pytest",
              *RESEARCH_ONLY, "torch", "torchvision", "torchaudio", "transformers", "tokenizers", "huggingface_hub",
              "hf_xet", "safetensors", "onnxruntime", "stable_baselines3", "gymnasium", "tensorboard", "PySide6.QtWebEngineCore",
              "PySide6.Qt3DCore", "PySide6.QtQuick", "PySide6.QtQml", "PySide6.QtMultimedia"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name="NiveshRL",
    icon=os.path.join(SPECPATH, "niveshrl.ico"),
    console=False,
    upx=False,
)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="NiveshRL")
