"""
========================================================================
  NEXUS — Flask Backend API  v2.1
  Build: 2026-03-27 final
  Clean rewrite based on Application.py (Streamlit -> REST)
========================================================================
  Run:
    python nexus_api2.py          ->  http://localhost:5000

  Place this file in the SAME folder as HMB_Spring_2026_Helpers.py
========================================================================
"""

# ── stdlib ────────────────────────────────────────────────────────────────────
import base64
import io
import json
import os
import pickle
import subprocess
import sys
import tempfile
import traceback
import uuid
import warnings

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

print("THIS DIR:", _THIS_DIR)

if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

print("SYS PATH:", sys.path[:3])

warnings.filterwarnings("ignore")

# ── auto-install missing packages ─────────────────────────────────────────────
def _ensure(import_name, pip_name):
    try:
        __import__(import_name)
    except ImportError:
        print(f"  [auto-install] {pip_name} ...", flush=True)
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", pip_name, "--quiet"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )

print("Checking dependencies ...", flush=True)
for _imp, _pip in [
    ("flask",      "flask"),
    ("flask_cors", "flask-cors"),
    ("cv2",        "opencv-python"),
    ("matplotlib", "matplotlib"),
    ("numpy",      "numpy"),
    ("pandas",     "pandas"),
    ("PIL",        "pillow"),
    ("sklearn",    "scikit-learn"),
    ("trimesh",    "trimesh"),
    ("optuna",     "optuna"),
    ("shutup",     "shutup"),
]:
    _ensure(_imp, _pip)
print("Dependencies OK.\n", flush=True)

# ── enable experimental sklearn features needed by HMB helpers ────────────────
try:
    from sklearn.experimental import enable_halving_search_cv  # noqa
except Exception:
    pass
try:
    from sklearn.experimental import enable_iterative_imputer  # noqa
except Exception:
    pass

# ── check TensorFlow availability ─────────────────────────────────────────────
try:
    import tensorflow as _tf_check  # noqa
    _TF_AVAILABLE = True
    print("[TF] TensorFlow found — all tabs enabled.", flush=True)
except ImportError:
    _TF_AVAILABLE = False
    print("[TF] TensorFlow NOT found — CNN tabs (3 & 4) will be disabled.", flush=True)

# ── third-party imports (safe after mocks) ────────────────────────────────────
import cv2                                        # noqa
import matplotlib; matplotlib.use("Agg")         # noqa
import matplotlib.pyplot as plt                   # noqa
import numpy as np                                # noqa
import pandas as pd                               # noqa
from flask import Flask, Response, jsonify, request, stream_with_context, send_from_directory  # noqa
from flask_cors import CORS                       # noqa
from PIL import Image                             # noqa

# ── import course helpers ─────────────────────────────────────────────────────
try:
    import HMB_Spring_2026_Helpers as HMB
    print("HMB LOADED SUCCESSFULLY")
    _HMB_OK = True
except Exception as e:
    print("HMB IMPORT ERROR:", e)
    _HMB_OK = False

# ── import HMB helpers ────────────────────────────────────────────────────────
# If TF is not installed, HMB_Spring_2026_Helpers.py will fail at the top-level
# "import tensorflow" line. We patch sys.modules with a minimal stub ONLY during
# the import, then remove our stubs so real TF calls are never intercepted.

##import types as _types
##
##def _patch_tf_for_import():
##    Insert minimal TF stubs into sys.modules so HMB helpers can be imported.
##
##    _patched = []
##
##    def _dummy_factory(name="DummyTF"):
##        class _Dummy:
##            def __init__(self, *args, **kwargs):
##                self._name = name
##            def __call__(self, *args, **kwargs):
##                return self
##            def __getattr__(self, _):
##                return self
##            def __iter__(self):
##                return iter(())
##            def __repr__(self):
##                return f"<{self._name}>"
##        return _Dummy
##
##    _base_exports = {
##        "tensorflow.keras.optimizers": ["Adam", "SGD", "RMSprop", "Adagrad", "Adadelta", "Adamax", "Nadam", "Ftrl"],
##        "tensorflow.keras.losses": ["BinaryCrossentropy", "CategoricalCrossentropy", "SparseCategoricalCrossentropy"],
##        "tensorflow.keras.metrics": ["AUC", "Precision", "Recall", "BinaryAccuracy", "CategoricalAccuracy", "TruePositives", "TrueNegatives", "FalsePositives", "FalseNegatives"],
##        "tensorflow.keras.callbacks": ["ModelCheckpoint", "EarlyStopping", "CSVLogger", "ReduceLROnPlateau", "Callback"],
##        "tensorflow.keras.models": ["Model", "Sequential", "load_model"],
##        "tensorflow.keras.layers": ["Dense", "Dropout", "Flatten", "Input", "BatchNormalization", "GlobalAveragePooling2D", "Conv2D", "MaxPooling2D"],
##        "tensorflow.keras.applications": ["MobileNetV2", "ResNet50", "VGG16", "EfficientNetB0", "InceptionV3"],
##    }
##
##    _tf_names = [
##        "tensorflow", "tensorflow.keras", "tensorflow.keras.models",
##        "tensorflow.keras.layers", "tensorflow.keras.callbacks",
##        "tensorflow.keras.optimizers", "tensorflow.keras.preprocessing",
##        "tensorflow.keras.preprocessing.image", "tensorflow.keras.applications",
##        "tensorflow.keras.applications.mobilenet_v2",
##        "tensorflow.keras.applications.resnet50",
##        "tensorflow.keras.applications.vgg16",
##        "tensorflow.keras.applications.efficientnet",
##        "tensorflow.keras.applications.inception_v3",
##        "tensorflow.keras.utils", "tensorflow.keras.regularizers",
##        "tensorflow.keras.metrics", "tensorflow.keras.losses",
##        "tensorflow.python", "tensorflow.python.framework",
##        "tensorflow.config", "tensorflow.data",
##    ]
##    for _n in _tf_names:
##        if _n not in sys.modules:
##            _mod = _types.ModuleType(_n)
##            _mod.__path__ = []
##            _exports = list(_base_exports.get(_n, []))
##            _mod.__all__ = _exports
##            for _name in _exports:
##                if _name == "load_model":
##                    setattr(_mod, _name, lambda *a, **kw: None)
##                else:
##                    setattr(_mod, _name, _dummy_factory(_name))
##            _mod.__getattr__ = lambda k, _factory=_dummy_factory: _factory(k)
##            sys.modules[_n] = _mod
##            _patched.append(_n)
##    return _patched
##
##def _unpatch_tf(patched_names):
##    Remove our stubs — real TF (if installed) stays, stubs are deleted.
##        for _n in patched_names:
##        # Only remove if it is still our stub (has no real __version__)
##        mod = sys.modules.get(_n)
##        if mod is not None and not hasattr(mod, "__version__"):
##            del sys.modules[_n]
##
##_patched_names = [] if _TF_AVAILABLE else _patch_tf_for_import()
##
##try:
from HMB_Spring_2026_Helpers import *
##    # Remove TF stubs NOW — after import, real function calls must not hit mocks
##    _unpatch_tf(_patched_names)
##
##    # Smoke test — calls the real FOF function (no TF involved)
##    _t = np.random.randint(0, 255, (32, 32), dtype=np.uint8)
##    _f, _ = FirstOrderFeatures2D(_t, isNorm=True, ignoreZeros=True)
##    print(f"HMB_Spring_2026_Helpers OK — FOF smoke test: {len(_f)} features.\n", flush=True)
##    _HMB_OK = True

##except Exception as _e:
##    _unpatch_tf(_patched_names)
##    _HMB_ERR = str(_e)
##    _HMB_OK  = False
##    print(f"\nWARNING: HMB helpers failed: {_e}\n", flush=True)
##    traceback.print_exc()
##
##    def _stub(*a, **kw):
##        raise RuntimeError(f"HMB_Spring_2026_Helpers unavailable: {_HMB_ERR}")
##
##    (CalculateGLCMCooccuranceMatrix, CalculateGLCMFeaturesOptimized,
##     CalculateGLRLMFeatures, CalculateGLRLMRunLengthMatrix,
##     CalculateGLSZMFeatures, CalculateGLSZMSizeZoneMatrix,
##     ExtractMultipleObjectsFromROI, FirstOrderFeatures2D,
##     MachineLearningClassificationV2) = (_stub,) * 9
# ── Flask app ─────────────────────────────────────────────────────────────────
app  = Flask(__name__)
CORS(app)
SESSION: dict = {}   # { uuid -> { "objects": ..., "type": ... } }


# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=110, facecolor="#0d0d14")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def _imread(path: str):
    """Handles paths with spaces/unicode on Windows."""
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)


def _extract_region_features(
    region, do_fof, do_glcm, do_glrlm, do_glszm,
    normalize, ignore_zeros,
    glcm_d, glcm_theta, glcm_sym, glrlm_theta, glszm_conn,
):
    """Direct replica of extract_features_from_region() in Application.py."""
    feat = {}

    if do_fof:
        fof, _ = FirstOrderFeatures2D(region, isNorm=normalize, ignoreZeros=ignore_zeros)
        feat.update({f"FOF_{k}": v for k, v in fof.items()})

    if do_glcm:
        for d in glcm_d:
            for td in glcm_theta:
                try:
                    co = CalculateGLCMCooccuranceMatrix(
                        region, d, np.radians(td),
                        isSymmetric=glcm_sym, isNorm=normalize, ignoreZeros=ignore_zeros,
                    )
                    gf = CalculateGLCMFeaturesOptimized(co)
                    feat.update({f"GLCM_D{d}_T{td}_{k}": v for k, v in gf.items()})
                except Exception:
                    pass

    if do_glrlm:
        for td in glrlm_theta:
            try:
                rl = CalculateGLRLMRunLengthMatrix(
                    region, np.radians(td), isNorm=normalize, ignoreZeros=ignore_zeros
                )
                rf = CalculateGLRLMFeatures(rl, region)
                feat.update({f"GLRLM_T{td}_{k}": v for k, v in rf.items()})
            except Exception:
                pass

    if do_glszm:
        for conn in glszm_conn:
            try:
                sz, _, N, Z = CalculateGLSZMSizeZoneMatrix(
                    region, connectivity=conn, isNorm=normalize, ignoreZeros=ignore_zeros,
                )
                sf = CalculateGLSZMFeatures(sz, region, N, Z)
                feat.update({f"GLSZM_C{conn}_{k}": v for k, v in sf.items()})
            except Exception:
                pass

    return feat


# ─────────────────────────────────────────────────────────────────────────────
#  SERVE FRONTEND
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    print("NEW INDEX ROUTE ACTIVE", flush=True)
    path = _THIS_DIR
    return send_from_directory(path, "Pipeline Application Website.html")

    if os.path.isfile(p):
            print(f"[INDEX] Serving {p}", flush=True)
            with open(p, "r", encoding="utf-8") as f:
                return f.read(), 200, {"Content-Type": "text/html; charset=utf-8"}

    # Not found — show helpful debug info
    print(f"[INDEX] nexus_platform.html not found. Searched: {candidates}", flush=True)
    files_in_dir = os.listdir(_THIS_DIR)
    return f"""<html><body style="font-family:monospace;padding:30px;background:#04040d;color:#00d4ff;">
        <h2>nexus_platform.html not found</h2>
        <p>Script directory: <b>{_THIS_DIR}</b></p>
        <p>Files there: {files_in_dir}</p>
        <p>Make sure nexus_platform.html is in the same folder as nexus_api.py</p>
        <p style="color:#2cb67d">API itself is working fine — all /extract_features, /train_ml etc. endpoints are live.</p>
        </body></html>""", 200

@app.route("/favicon.ico")
def favicon():
    return "", 204

@app.route("/health", methods=["GET", "POST"])
def health():
    return jsonify({"status": "ok", "version": "2.0",
                    "hmb_loaded": _HMB_OK, "tf_available": _TF_AVAILABLE})


# ─────────────────────────────────────────────────────────────────────────────
#  TAB 1 — FEATURE EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/extract_features", methods=["POST"])
def extract_features():
    body = request.get_json(force=True)

    dataset_dir  = body.get("dataset_dir",  "").strip()
    output_csv   = body.get("output_csv",   "extracted_features.csv")
    target_size  = int(body.get("target_size",  128))
    ignore_zeros = bool(body.get("ignore_zeros", True))
    normalize    = bool(body.get("normalize",    True))
    do_fof       = bool(body.get("do_fof",   True))
    do_glcm      = bool(body.get("do_glcm",  False))
    do_glrlm     = bool(body.get("do_glrlm", False))
    do_glszm     = bool(body.get("do_glszm", False))
    glcm_d       = body.get("glcm_d",      [1])
    glcm_theta   = body.get("glcm_theta",  [0])
    glcm_sym     = bool(body.get("glcm_sym", False))
    glrlm_theta  = body.get("glrlm_theta", [0])
    glszm_conn   = body.get("glszm_conn",  [4])

    if not dataset_dir or not os.path.isdir(dataset_dir):
        return jsonify({"error": f"Dataset directory not found: '{dataset_dir}'"}), 400
    if not any([do_fof, do_glcm, do_glrlm, do_glszm]):
        return jsonify({"error": "Select at least one feature type."}), 400

    IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}

    def _stream():
        try:
            class_dirs = sorted([
                d for d in os.listdir(dataset_dir)
                if os.path.isdir(os.path.join(dataset_dir, d))
            ])
            if not class_dirs:
                yield 'ERROR:No class subfolders found.\n'; return

            print(f"[FE] Classes: {class_dirs}", flush=True)

            total_images = sum(
                1 for cls in class_dirs
                for f in os.listdir(os.path.join(dataset_dir, cls))
                if os.path.splitext(f)[1].lower() in IMG_EXTS and "_mask" not in f.lower()
            )
            print(f"[FE] Total images: {total_images}", flush=True)
            yield f"TOTAL:{total_images}\n"

            history, processed, skipped, done_count = [], 0, 0, 0
            _sample_before_b64 = _sample_after_b64 = None
            _sample_saved = False

            for cls in class_dirs:
                cls_path  = os.path.join(dataset_dir, cls)
                all_files = sorted(os.listdir(cls_path))
                img_files = [
                    f for f in all_files
                    if os.path.splitext(f)[1].lower() in IMG_EXTS
                    and "_mask" not in f.lower()
                ]

                for img_name in img_files:
                    stem, ext = os.path.splitext(img_name)
                    img_path  = os.path.join(cls_path, img_name)
                    mask_path = os.path.join(cls_path, f"{stem}_mask{ext}")

                    img_arr  = _imread(img_path)
                    if img_arr is None:
                        skipped += 1; done_count += 1; continue

                    mask_arr = _imread(mask_path) if os.path.exists(mask_path) \
                               else np.ones_like(img_arr) * 255

                    try:
                        if img_arr.shape[0] != target_size or img_arr.shape[1] != target_size:
                            img_arr_rs  = cv2.resize(img_arr,  (target_size, target_size), interpolation=cv2.INTER_AREA)
                            mask_arr_rs = cv2.resize(mask_arr, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
                        else:
                            img_arr_rs, mask_arr_rs = img_arr, mask_arr

                        regions = ExtractMultipleObjectsFromROI(
                            img_arr_rs, mask_arr_rs,
                            targetSize=(target_size, target_size),
                            cntAreaThreshold=0, sortByX=True,
                        )
                        region_to_use = regions[0] if regions else img_arr_rs

                        feat = _extract_region_features(
                            region_to_use, do_fof, do_glcm, do_glrlm, do_glszm,
                            normalize, ignore_zeros,
                            glcm_d, glcm_theta, glcm_sym, glrlm_theta, glszm_conn,
                        )
                        if feat:
                            history.append({"File": img_name, **feat, "Class": cls})
                            processed += 1
                            if not _sample_saved:
                                try:
                                    _, bb = cv2.imencode(".png", cv2.cvtColor(img_arr, cv2.COLOR_GRAY2BGR) if img_arr.ndim==2 else img_arr)
                                    _, ab = cv2.imencode(".png", cv2.cvtColor(region_to_use, cv2.COLOR_GRAY2BGR) if region_to_use.ndim==2 else region_to_use)
                                    _sample_before_b64 = base64.b64encode(bb).decode()
                                    _sample_after_b64  = base64.b64encode(ab).decode()
                                    _sample_saved = True
                                except Exception: pass
                        else:
                            skipped += 1

                    except MemoryError:
                        skipped += 1
                        print(f"[FE] MemoryError {cls}/{img_name}", flush=True)
                    except Exception as ex:
                        skipped += 1
                        if skipped <= 5:
                            print(f"[FE] Error {cls}/{img_name}: {ex}", flush=True)

                    done_count += 1
                    # Send progress every 10 images — keeps browser connection alive
                    if done_count % 5 == 0 or done_count == total_images:
                        pct = round(done_count / max(total_images,1) * 100, 1)
                        print(f"[FE] PROGRESS:{done_count}/{total_images}:{pct}:{cls}:{processed}:{skipped}", flush=True)
                        yield f"PROGRESS:{done_count}/{total_images}:{pct}:{cls}:{processed}:{skipped}\n"

            if not history:
                yield 'ERROR:No features extracted. Check dataset structure.\n'; return

            feat_df      = pd.DataFrame(history)
            feature_cols = [c for c in feat_df.columns if c not in ("File", "Class")]

            csv_save_path = os.path.join(dataset_dir, output_csv)
            feat_df.to_csv(csv_save_path, index=False)
            print(f"[FE] CSV saved: {csv_save_path}", flush=True)

            preview_df = feat_df.head(5).copy()
            for c in preview_df.select_dtypes(include=[np.floating]).columns:
                preview_df[c] = preview_df[c].round(5)

            sample_csv_b64 = base64.b64encode(feat_df.to_csv(index=False).encode()).decode()
            key = str(uuid.uuid4())
            SESSION[key] = {"feat_df": feat_df, "type": "features"}

            resp = {
                "processed":      processed,
                "skipped":        skipped,
                "classes":        class_dirs,
                "total_rows":     len(feat_df),
                "total_features": len(feature_cols),
                "preview":        preview_df.to_dict(orient="records"),
                "csv_b64":        sample_csv_b64,
                "csv_path":       csv_save_path,
                "session_key":    key,
            }
            if _sample_before_b64: resp["sample_before_b64"] = _sample_before_b64
            if _sample_after_b64:  resp["sample_after_b64"]  = _sample_after_b64

            yield "RESULTS_JSON:" + json.dumps(resp) + "\n"

        except Exception as ex:
            traceback.print_exc()
            yield f"ERROR:{str(ex)}\n"

    return Response(stream_with_context(_stream()), mimetype="text/plain",
                    headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"})


@app.route("/debug_dataset", methods=["POST"])
def debug_dataset():
    try:
        body = request.get_json(force=True)
        dataset_dir = str(body.get("dataset_dir", "")).strip()
        if not dataset_dir or not os.path.isdir(dataset_dir):
            return jsonify({"error": f"Dataset directory not found: '{dataset_dir}'"}), 400
        if not _HMB_OK:
            return jsonify({"error": f"HMB helpers unavailable: {_HMB_ERR}"}), 500

        img_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}
        class_dirs = sorted([
            d for d in os.listdir(dataset_dir)
            if os.path.isdir(os.path.join(dataset_dir, d))
        ])
        if not class_dirs:
            return jsonify({"error": "No class subfolders found."}), 400

        report = []
        for cls in class_dirs:
            cls_path = os.path.join(dataset_dir, cls)
            all_files = sorted(os.listdir(cls_path))
            plain_images = [
                f for f in all_files
                if os.path.splitext(f)[1].lower() in img_exts and "_mask" not in f.lower()
            ]
            mask_files = [f for f in all_files if "_mask" in f.lower()]
            cls_info = {
                "name": cls,
                "plain_images": len(plain_images),
                "mask_files": len(mask_files),
                "sample_images": plain_images[:5],
                "decode_tests": [],
                "roi_test": None,
                "fof_test": None,
            }

            for fname in plain_images[:3]:
                path = os.path.join(cls_path, fname)
                img = _imread(path)
                cls_info["decode_tests"].append({
                    "file": fname,
                    "readable": img is not None,
                    "shape": list(img.shape) if img is not None else None,
                })

            if plain_images:
                sample_name = plain_images[0]
                sample_path = os.path.join(cls_path, sample_name)
                stem, ext = os.path.splitext(sample_name)
                mask_path = os.path.join(cls_path, f"{stem}_mask{ext}")
                sample_img = _imread(sample_path)
                sample_mask = _imread(mask_path) if os.path.exists(mask_path) else None
                if sample_img is not None:
                    if sample_mask is None:
                        sample_mask = np.ones_like(sample_img) * 255
                    try:
                        _, mask_bin = cv2.threshold(sample_mask, 10, 255, cv2.THRESH_BINARY)
                        _dbg_sz = int(body.get("target_size", 128))
                        regs = ExtractMultipleObjectsFromROI(sample_img, mask_bin, targetSize=(_dbg_sz, _dbg_sz), cntAreaThreshold=0, sortByX=True)
                        cls_info["roi_test"] = {"regions_found": len(regs)}
                        if regs:
                            fof, _ = FirstOrderFeatures2D(regs[0], isNorm=True, ignoreZeros=True)
                            cls_info["fof_test"] = {"feature_count": len(fof)}
                        else:
                            cls_info["fof_test"] = {"feature_count": 0, "error": "No regions extracted."}
                    except Exception as ex:
                        cls_info["roi_test"] = {"regions_found": 0, "error": str(ex)}
                        cls_info["fof_test"] = {"feature_count": 0, "error": str(ex)}
            report.append(cls_info)

        return jsonify({
            "dataset_dir": dataset_dir,
            "class_count": len(class_dirs),
            "classes": report,
        })

    except Exception as ex:
        traceback.print_exc()
        return jsonify({"error": str(ex), "traceback": traceback.format_exc()}), 500


# ─────────────────────────────────────────────────────────────────────────────
#  TAB 2 — ML CLASSIFICATION
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/train_ml", methods=["POST"])
def train_ml():
    try:
        train_file = request.files.get("train_csv")
        if not train_file:
            return jsonify({"error": "No training CSV uploaded."}), 400

        test_file     = request.files.get("test_csv")
        target_col    = request.form.get("target_column",     "Class")
        drop_first    = request.form.get("drop_first",        "true").lower() == "true"
        test_ratio    = float(request.form.get("test_ratio",  0.2))
        model_choice  = request.form.get("model",             "MLP")
        scaler_choice = request.form.get("scaler",            "") or None
        fs_tech       = request.form.get("feature_selection", "") or None
        fs_ratio      = int(request.form.get("fs_ratio",      80))

        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            tmp.write(train_file.read())
            train_path = tmp.name

        test_path = None
        if test_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                tmp.write(test_file.read())
                test_path = tmp.name

        print(f"[ML] {model_choice} scaler={scaler_choice} fs={fs_tech}", flush=True)

        metrics, plt_obj, objects = MachineLearningClassificationV2(
            train_path, scaler_choice, model_choice, fs_tech, fs_ratio,
            testRatio=test_ratio, testFilePath=test_path,
            targetColumn=target_col, dropFirstColumn=drop_first,
        )
        print("[ML] Training complete.", flush=True)

        cm_b64 = None
        if plt_obj is not None:
            try:
                cm_b64 = _fig_to_b64(plt_obj)
                plt.close("all")
            except Exception:
                pass

        pkl_b64 = base64.b64encode(pickle.dumps(objects)).decode()
        # Also store the target_size used during training inside objects
        # so prediction can use the exact same resize dimensions
        # Store the target_size used during feature extraction so predict_ml uses the same
        # Frontend sends target_size_used; fall back to reading from CSV column names if missing
        _ts = request.form.get("target_size_used", "").strip()
        objects["_target_size"] = int(_ts) if _ts.isdigit() else 128
        key     = str(uuid.uuid4())
        SESSION[key] = {"objects": objects, "type": "ml_model"}

        scalar_metrics = {}
        for k in ["Macro Accuracy","Macro Precision","Macro Recall","Macro F1",
                  "Weighted Accuracy","Weighted Precision","Weighted Recall","Weighted F1",
                  "Micro Accuracy"]:
            if k in metrics and np.isscalar(metrics[k]):
                scalar_metrics[k] = float(metrics[k])

        for p in [train_path, test_path]:
            try:
                if p: os.unlink(p)
            except Exception:
                pass

        return jsonify({
            "metrics":              scalar_metrics,
            "pkl_b64":              pkl_b64,
            "confusion_matrix_b64": cm_b64,
            "session_key":          key,
        })

    except Exception as ex:
        traceback.print_exc()
        return jsonify({"error": str(ex)}), 500


# ─────────────────────────────────────────────────────────────────────────────
#  TAB 3 — DEEP LEARNING (subprocess via be645)
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/train_cnn", methods=["POST"])
def train_cnn():
    try:
        body          = request.get_json(force=True)
        be645_python  = body.get("be645_python",  "").strip()
        dataset_dir   = body.get("dataset_dir",   "").strip()
        output_dir    = body.get("output_dir",    "./CNN_Output").strip()
        helpers_dir   = body.get("helpers_dir",   "").strip() or _THIS_DIR
        base_model    = body.get("base_model",    "MobileNetV2")
        img_h         = int(body.get("img_h",     256))
        img_w         = int(body.get("img_w",     256))
        batch_size    = int(body.get("batch_size", 32))
        epochs        = int(body.get("epochs",    50))
        patience      = int(body.get("patience",  10))
        lr            = float(body.get("lr",      1e-3))

        errors = []
        if not be645_python or not os.path.isfile(be645_python):
            errors.append(f"be645 Python not found: '{be645_python}'")
        if not dataset_dir or not os.path.isdir(dataset_dir):
            errors.append(f"Dataset dir not found: '{dataset_dir}'")
        if errors:
            return jsonify({"error": "\n".join(errors)}), 400

        os.makedirs(output_dir, exist_ok=True)

        script = f'''
import os, sys, json, warnings
warnings.filterwarnings("ignore")
BASE_DIR = r"{helpers_dir}"
sys.path.insert(0, BASE_DIR)
import HMB_Spring_2026_Helpers as HMB
from HMB_Spring_2026_Helpers import *
os.makedirs(r"{output_dir}", exist_ok=True)

import numpy as np, pandas as pd, tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2,ResNet50,VGG16,EfficientNetB0,InceptionV3
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping,CSVLogger,ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

print("TF:", tf.__version__, "GPUs:", len(tf.config.list_physical_devices("GPU")))

dataset_dir,output_dir,base_model_name = r"{dataset_dir}",r"{output_dir}","{base_model}"
input_shape = ({img_h},{img_w},3)
batch_size,epochs,patience,lr = {batch_size},{epochs},{patience},{lr}

file_paths,labels=[],[]
for c in os.listdir(dataset_dir):
    cp=os.path.join(dataset_dir,c)
    if os.path.isdir(cp):
        for f in os.listdir(cp):
            if os.path.splitext(f)[1].lower() in [".png",".jpg",".jpeg",".bmp"] and "_mask" not in f.lower():
                file_paths.append(os.path.join(cp,f)); labels.append(c)

df=pd.DataFrame({{"image_path":file_paths,"label":labels}})
n_classes=len(df["label"].unique())
print(f"Dataset: {{len(df)}} images, {{n_classes}} classes")

train_df,test_df=train_test_split(df,test_size=0.2,random_state=42,stratify=df["label"])
train_df,val_df =train_test_split(train_df,test_size=0.25,random_state=42,stratify=train_df["label"])

aug=ImageDataGenerator(rescale=1./255,rotation_range=10,width_shift_range=0.15,
    height_shift_range=0.15,shear_range=0.15,zoom_range=0.15,horizontal_flip=True,fill_mode="nearest")
train_gen=aug.flow_from_dataframe(train_df,x_col="image_path",y_col="label",
    target_size=input_shape[:2],batch_size=batch_size,class_mode="categorical",shuffle=True)
val_gen=ImageDataGenerator(rescale=1./255).flow_from_dataframe(val_df,x_col="image_path",y_col="label",
    target_size=input_shape[:2],batch_size=batch_size,class_mode="categorical",shuffle=False)
test_gen=ImageDataGenerator(rescale=1./255).flow_from_dataframe(test_df,x_col="image_path",y_col="label",
    target_size=input_shape[:2],batch_size=batch_size,class_mode="categorical",shuffle=False)

bmap={{"MobileNetV2":MobileNetV2,"ResNet50":ResNet50,"VGG16":VGG16,"EfficientNetB0":EfficientNetB0,"InceptionV3":InceptionV3}}
base=bmap[base_model_name](include_top=False,weights="imagenet",input_shape=input_shape)
base.trainable=False
x=layers.GlobalAveragePooling2D()(base.output)
x=layers.BatchNormalization()(x)
x=layers.Dense(256,activation="relu")(x)
x=layers.Dropout(0.5)(x)
out=layers.Dense(n_classes,activation="softmax")(x)
model=models.Model(inputs=base.input,outputs=out)
model.compile(optimizer=Adam(lr),loss="categorical_crossentropy",metrics=["accuracy"])

model.fit(train_gen,validation_data=val_gen,epochs=epochs,callbacks=[
    ModelCheckpoint(os.path.join(output_dir,"Model.keras"),save_best_only=True,monitor="val_accuracy",verbose=1),
    EarlyStopping(monitor="val_accuracy",patience=patience,restore_best_weights=True),
    CSVLogger(os.path.join(output_dir,"Log.csv")),
    ReduceLROnPlateau(monitor="val_loss",factor=0.5,patience=patience//2,min_lr=1e-7),
],verbose=1)

with open(os.path.join(output_dir,"class_indices.json"),"w") as f:
    json.dump({{"class_indices":train_gen.class_indices,"input_shape":list(input_shape)}},f)

best=tf.keras.models.load_model(os.path.join(output_dir,"Model.keras"))
preds=best.predict(test_gen,verbose=0); y_pred=np.argmax(preds,axis=1); y_true=test_gen.classes
results={{"Test Accuracy":float(accuracy_score(y_true,y_pred)),
          "Precision":float(precision_score(y_true,y_pred,average="macro",zero_division=0)),
          "Recall":float(recall_score(y_true,y_pred,average="macro",zero_division=0)),
          "F1":float(f1_score(y_true,y_pred,average="macro",zero_division=0))}}
print("RESULTS_JSON:"+json.dumps(results))
print("TRAINING_COMPLETE")
'''
        script_path = os.path.join(output_dir, "_cnn_train.py")
        with open(script_path, "w") as f:
            f.write(script)

        def _stream():
            proc = subprocess.Popen(
                [be645_python, script_path],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
            )
            for line in proc.stdout:
                yield line
            proc.wait()
            if proc.returncode != 0:
                yield f"ERROR: exited with code {proc.returncode}\n"

        return Response(stream_with_context(_stream()), mimetype="text/plain",
                        headers={"X-Accel-Buffering": "no"})

    except Exception as ex:
        traceback.print_exc()
        return jsonify({"error": str(ex)}), 500


# ─────────────────────────────────────────────────────────────────────────────
#  TAB 4 — PREDICT (ML)
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/predict_ml", methods=["POST"])
def predict_ml():
    try:
        objects     = None
        session_key = request.form.get("session_key", "")
        if session_key and session_key in SESSION:
            objects = SESSION[session_key]["objects"]
        else:
            pkl_file = request.files.get("model_pkl")
            if not pkl_file:
                return jsonify({"error": "No ML model provided."}), 400
            try:
                objects = pickle.loads(pkl_file.read())
            except Exception as pkl_err:
                err_str = str(pkl_err)
                if "BitGenerator" in err_str or "numpy" in err_str.lower():
                    return jsonify({"error":
                        "Model pickle is incompatible with the current NumPy version. "
                        "This happens when the .pkl was saved in a different environment. "
                        "Solution: retrain the model in the Ui environment (Tab 2) and "
                        "use the session model or re-download the new .pkl."
                    }), 400
                return jsonify({"error": f"Could not load model: {pkl_err}"}), 400

        img_file = request.files.get("image")
        if not img_file:
            return jsonify({"error": "No image uploaded."}), 400

        img = cv2.imdecode(np.frombuffer(img_file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return jsonify({"error": "Could not decode image."}), 400

        print(f"[PRED] Image shape: {img.shape}", flush=True)

        mask_file = request.files.get("mask")
        if mask_file:
            # User provided a mask — use it directly
            mask = cv2.imdecode(np.frombuffer(mask_file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
            print(f"[PRED] Using uploaded mask", flush=True)
        else:
            # No mask uploaded — use full white mask, EXACTLY like training did
            # Training used: mask_arr = np.ones_like(img_arr) * 255 when no mask file existed
            mask = np.ones_like(img) * 255
            print(f"[PRED] No mask — using full white mask (same as training fallback)", flush=True)

        # Use the target_size stored in the pkl from training time.
        # If missing (old pkl), try to infer from the request, then fall back to 128.
        _ts_override = request.form.get("target_size", "").strip()
        if _ts_override.isdigit():
            pred_target_size = int(_ts_override)
        else:
            pred_target_size = objects.get("_target_size", None)
            if pred_target_size is None:
                # Last resort: use the smaller image dimension as a reasonable default
                pred_target_size = min(img.shape[:2])
                pred_target_size = min(pred_target_size, 512)  # cap at 512
                print(f"[PRED] _target_size missing from pkl — inferred {pred_target_size} from image", flush=True)
        pred_target_size = int(pred_target_size)
        print(f"[PRED] Using target_size={pred_target_size}", flush=True)
        try:
            _, mask_bin = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
            regs = ExtractMultipleObjectsFromROI(
                img, mask_bin,
                targetSize=(pred_target_size, pred_target_size),
                cntAreaThreshold=0,
                sortByX=True,
            )
            if regs:
                region = regs[0]
                print(f"[PRED] ROI extracted: {region.shape}", flush=True)
            else:
                region = cv2.resize(img, (pred_target_size, pred_target_size))
                print(f"[PRED] No ROI found — using plain resize", flush=True)
        except Exception as roi_err:
            region = cv2.resize(img, (pred_target_size, pred_target_size))
            print(f"[PRED] ROI extraction failed ({roi_err}) — using plain resize", flush=True)

        # ── Unpack model objects ───────────────────────────────────────────────
        # Log ALL keys in the pkl so we know exactly what was saved
        print(f"[PRED] pkl keys: {list(objects.keys())}", flush=True)

        model  = objects["Model"]
        scaler = objects.get("Scaler")
        le     = objects["LabelEncoder"]
        fs     = objects.get("FeatureSelector")

        # CurrentColumns holds the exact feature columns used during training
        # Try multiple key names without using pandas objects in boolean context
        cols = None
        for key in ["CurrentColumns", "Columns", "FeatureColumns", "Features", "X_columns"]:
            val = objects.get(key)
            if val is not None:
                cols = list(val)
                break

        # SelectedFeatures is an alternative that some versions store
        selected = objects.get("SelectedFeatures")
        if selected is None:
            selected = objects.get("selected_features")

        print(f"[PRED] cols={cols[:5] if cols is not None else None}... len={len(cols) if cols is not None else 0}", flush=True)
        print(f"[PRED] classes={list(le.classes_)}", flush=True)
        print(f"[PRED] scaler={scaler}", flush=True)
        print(f"[PRED] fs={fs}", flush=True)

        # ── Detect which feature families are in the training columns ─────────
        import re as _re

        do_fof   = True
        do_glcm  = False
        do_glrlm = False
        do_glszm = False
        glcm_d, glcm_theta, glcm_sym = [1], [0], False
        glrlm_theta = [0]
        glszm_conn  = [4]

        if cols is not None and len(cols) > 0:
            cols = list(cols)
            col_str = " ".join(cols)
            do_fof   = any(c.startswith("FOF_")   for c in cols)
            do_glcm  = any(c.startswith("GLCM_")  for c in cols)
            do_glrlm = any(c.startswith("GLRLM_") for c in cols)
            do_glszm = any(c.startswith("GLSZM_") for c in cols)

            if do_glcm:
                ds = sorted(set(int(m) for m in _re.findall(r"GLCM_D(\d+)_",      col_str)))
                ts = sorted(set(int(m) for m in _re.findall(r"GLCM_D\d+_T(\d+)_", col_str)))
                glcm_d, glcm_theta = ds or [1], ts or [0]
            if do_glrlm:
                ts = sorted(set(int(m) for m in _re.findall(r"GLRLM_T(\d+)_", col_str)))
                glrlm_theta = ts or [0]
            if do_glszm:
                cs = sorted(set(int(m) for m in _re.findall(r"GLSZM_C(\d+)_", col_str)))
                glszm_conn = cs or [4]

        print(f"[PRED] Families: fof={do_fof} glcm={do_glcm}(d={glcm_d},t={glcm_theta}) glrlm={do_glrlm} glszm={do_glszm}", flush=True)

        # ── Extract features matching training ────────────────────────────────
        feat = _extract_region_features(
            region,
            do_fof, do_glcm, do_glrlm, do_glszm,
            normalize=True, ignore_zeros=True,
            glcm_d=glcm_d, glcm_theta=glcm_theta, glcm_sym=glcm_sym,
            glrlm_theta=glrlm_theta, glszm_conn=glszm_conn,
        )
        feat_series = pd.Series(feat)
        print(f"[PRED] Extracted {len(feat_series)} features. Keys sample: {list(feat_series.index[:5])}", flush=True)

        # ── Align to training columns exactly ─────────────────────────────────
        feat_df = pd.DataFrame([feat_series])
        if cols is not None and len(cols) > 0:
            missing = [c for c in cols if c not in feat_df.columns]
            if missing:
                print(f"[PRED] WARNING: {len(missing)} missing cols, filling with 0: {missing[:5]}", flush=True)
            for c in cols:
                if c not in feat_df.columns:
                    feat_df[c] = 0.0
            feat_df = feat_df[cols]
        else:
            # No column list — use whatever we extracted (risky but best we can do)
            print("[PRED] WARNING: No CurrentColumns in pkl — using raw extracted features", flush=True)

        print(f"[PRED] Feature matrix shape: {feat_df.shape}", flush=True)

        X = feat_df.values
        if scaler is not None:
            X = scaler.transform(X)
            print(f"[PRED] After scaling: mean={X.mean():.4f}", flush=True)
        if fs is not None:
            X = fs.transform(X)
            print(f"[PRED] After FS: shape={X.shape}", flush=True)

        # Raw scores for all classes
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)[0]
            probs_dict = {le.classes_[i]: float(probs[i]) for i in range(len(le.classes_))}
            pred_label = le.classes_[int(np.argmax(probs))]
            print(f"[PRED] Probabilities: {probs_dict}", flush=True)
        else:
            pred_label = le.inverse_transform([model.predict(X)[0]])[0]
            probs_dict = {pred_label: 1.0}

        print(f"[PRED] Final result: {pred_label}", flush=True)
        return jsonify({"predicted_class": pred_label, "probabilities": probs_dict})

    except Exception as ex:
        traceback.print_exc()
        return jsonify({"error": str(ex)}), 500


# ─────────────────────────────────────────────────────────────────────────────
#  TAB 4 — PREDICT (CNN)
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/predict_cnn", methods=["POST"])
def predict_cnn():
    if not _TF_AVAILABLE:
        return jsonify({"error": "TensorFlow not installed. CNN prediction unavailable."}), 400
    try:
        import tensorflow as tf

        input_hw        = int(request.form.get("input_size", 256))
        class_labels_raw = request.form.get("class_labels", "")
        model_path      = request.form.get("model_path", "").strip()
        ci_path         = request.form.get("ci_path",    "").strip()
        keras_file      = request.files.get("model_keras")
        cnn_model       = None
        class_indices   = {}

        if keras_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".keras") as tmp:
                tmp.write(keras_file.read())
                cnn_model = tf.keras.models.load_model(tmp.name)
        elif model_path and os.path.isfile(model_path):
            cnn_model = tf.keras.models.load_model(model_path)
        else:
            return jsonify({"error": "No CNN model provided."}), 400

        if ci_path and os.path.isfile(ci_path):
            with open(ci_path) as f:
                ci_data = json.load(f)
            class_indices = ci_data.get("class_indices", {})
            # input_shape stored as [H, W, C] — use H as the resize target
            _shape = ci_data.get("input_shape", None)
            if _shape and len(_shape) >= 2:
                input_hw = int(_shape[0])   # use stored H from training
            # If model has a known input shape, use that as ground truth
            try:
                model_input = cnn_model.input_shape  # (None, H, W, C)
                if model_input and len(model_input) == 4:
                    input_hw = int(model_input[1])
            except Exception:
                pass
        elif class_labels_raw:
            labels = [l.strip() for l in class_labels_raw.split(",") if l.strip()]
            class_indices = {l: i for i, l in enumerate(labels)}

        img_file = request.files.get("image")
        if not img_file:
            return jsonify({"error": "No image uploaded."}), 400

        pil_img = Image.open(img_file).convert("RGB").resize((input_hw, input_hw))
        img_arr = np.expand_dims(np.array(pil_img, dtype=np.float32) / 255.0, 0)

        preds    = cnn_model.predict(img_arr, verbose=0)[0]
        pred_idx = int(np.argmax(preds))
        idx2cls  = {v: k for k, v in class_indices.items()} if class_indices else {}

        return jsonify({
            "predicted_class": idx2cls.get(pred_idx, f"Class {pred_idx}"),
            "confidence":      float(preds[pred_idx]) * 100,
            "probabilities":   {idx2cls.get(i, f"Class {i}"): float(preds[i]) for i in range(len(preds))},
        })

    except Exception as ex:
        traceback.print_exc()
        return jsonify({"error": str(ex)}), 500


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

    print("=" * 55)
    print("  NEXUS Flask API v2.3  —  http://localhost:5000")
    print("=" * 55)

    # Try waitress first (no request timeout, production-grade)
    # If not installed, fall back to Flask dev server with extended timeout
    try:
        from waitress import serve
        print("  Using waitress WSGI server (no timeout limits)", flush=True)
        serve(app, host="127.0.0.1", port=5000, threads=4,
              channel_timeout=3600,   # 1 hour max per request
              cleanup_interval=30)
    except ImportError:
        print("  waitress not found — installing...", flush=True)
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "waitress", "--quiet"],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            from waitress import serve
            print("  Using waitress WSGI server", flush=True)
            serve(app, host="127.0.0.1", port=5000, threads=4, channel_timeout=3600)
        except Exception:
            print("  Falling back to Flask dev server", flush=True)
            app.run(host="127.0.0.1", port=5000, debug=False, threaded=True, use_reloader=False)
