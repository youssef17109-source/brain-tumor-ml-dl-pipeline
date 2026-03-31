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
    ("shap",       "shap"),
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
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500 MB — covers large .keras model uploads
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
    return send_from_directory(_THIS_DIR, "Pipeline Application.HTML")

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
    target_size  = int(body.get("target_size", 128))
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
            class_dirs = sorted([d for d in os.listdir(dataset_dir)
                                  if os.path.isdir(os.path.join(dataset_dir, d))])
            if not class_dirs:
                yield "ERROR:No class subfolders found.\n"; return

            print(f"[FE] ══════════════════════════════════════", flush=True)
            print(f"[FE] TARGET SIZE  : {target_size} x {target_size} px", flush=True)
            print(f"[FE] FEATURE TYPES: FOF={do_fof} GLCM={do_glcm} GLRLM={do_glrlm} GLSZM={do_glszm}", flush=True)
            print(f"[FE] Classes: {class_dirs}", flush=True)
            total_images = sum(1 for cls in class_dirs
                               for f in os.listdir(os.path.join(dataset_dir, cls))
                               if os.path.splitext(f)[1].lower() in IMG_EXTS and "_mask" not in f.lower())
            print(f"[FE] Total images: {total_images}", flush=True)
            yield f"TOTAL:{total_images}\n"

            history, processed, skipped, done_count = [], 0, 0, 0
            _sample_before_b64 = _sample_after_b64 = None
            _sample_saved = False

            for cls in class_dirs:
                cls_path  = os.path.join(dataset_dir, cls)
                img_files = [f for f in sorted(os.listdir(cls_path))
                             if os.path.splitext(f)[1].lower() in IMG_EXTS and "_mask" not in f.lower()]

                for img_name in img_files:
                    stem, ext = os.path.splitext(img_name)
                    img_arr   = _imread(os.path.join(cls_path, img_name))
                    if img_arr is None:
                        skipped += 1; done_count += 1; continue

                    mask_path = os.path.join(cls_path, f"{stem}_mask{ext}")
                    mask_arr  = _imread(mask_path) if os.path.exists(mask_path) else np.ones_like(img_arr)*255

                    try:
                        img_rs   = cv2.resize(img_arr,  (target_size, target_size), interpolation=cv2.INTER_AREA)
                        mask_rs  = cv2.resize(mask_arr, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
                        assert img_rs.shape[0] == target_size and img_rs.shape[1] == target_size,                             f"Resize failed: got {img_rs.shape} expected ({target_size},{target_size})"
                        regions  = ExtractMultipleObjectsFromROI(img_rs, mask_rs,
                                       targetSize=(target_size, target_size), cntAreaThreshold=0, sortByX=True)
                        region   = regions[0] if regions else img_rs
                        feat     = _extract_region_features(region, do_fof, do_glcm, do_glrlm, do_glszm,
                                       normalize, ignore_zeros, glcm_d, glcm_theta, glcm_sym, glrlm_theta, glszm_conn)
                        if feat:
                            history.append({"File": img_name, **feat, "Class": cls})
                            processed += 1
                            if not _sample_saved:
                                try:
                                    _, bb = cv2.imencode(".png", cv2.cvtColor(img_arr, cv2.COLOR_GRAY2BGR) if img_arr.ndim==2 else img_arr)
                                    _, ab = cv2.imencode(".png", cv2.cvtColor(region, cv2.COLOR_GRAY2BGR) if region.ndim==2 else region)
                                    _sample_before_b64 = base64.b64encode(bb).decode()
                                    _sample_after_b64  = base64.b64encode(ab).decode()
                                    _sample_saved = True
                                except Exception: pass
                        else:
                            skipped += 1
                    except MemoryError:
                        skipped += 1; print(f"[FE] MemoryError {cls}/{img_name}", flush=True)
                    except Exception as ex:
                        skipped += 1
                        if skipped <= 5: print(f"[FE] Error {cls}/{img_name}: {ex}", flush=True)

                    done_count += 1
                    if True:  # send every single image
                        pct = round(done_count / max(total_images,1) * 100, 1)
                        print(f"[FE] PROGRESS:{done_count}/{total_images}:{pct}:{cls}:{processed}:{skipped}", flush=True)
                        yield f"PROGRESS:{done_count}/{total_images}:{pct}:{cls}:{processed}:{skipped}\n"

            if not history:
                yield "ERROR:No features extracted. Check dataset structure.\n"; return

            feat_df      = pd.DataFrame(history)
            feature_cols = [c for c in feat_df.columns if c not in ("File","Class")]
            csv_save_path = os.path.join(dataset_dir, output_csv)
            feat_df.to_csv(csv_save_path, index=False)
            print(f"[FE] CSV saved: {csv_save_path}", flush=True)

            preview_df = feat_df.head(5).copy()
            for c in preview_df.select_dtypes(include=[np.floating]).columns:
                preview_df[c] = preview_df[c].round(5)

            csv_b64 = base64.b64encode(feat_df.to_csv(index=False).encode()).decode()
            key = str(uuid.uuid4())
            SESSION[key] = {"feat_df": feat_df, "type": "features"}

            resp = {"processed": processed, "skipped": skipped, "classes": class_dirs,
                    "total_rows": len(feat_df), "total_features": len(feature_cols),
                    "preview": preview_df.to_dict(orient="records"),
                    "csv_b64": csv_b64, "csv_path": csv_save_path, "session_key": key}
            if _sample_before_b64: resp["sample_before_b64"] = _sample_before_b64
            if _sample_after_b64:  resp["sample_after_b64"]  = _sample_after_b64
            yield "RESULTS_JSON:" + json.dumps(resp) + "\n"

        except Exception as ex:
            traceback.print_exc()
            yield f"ERROR:{str(ex)}\n"

    return Response(stream_with_context(_stream()), mimetype="text/plain",
                    headers={"X-Accel-Buffering":"no","Cache-Control":"no-cache"})


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
        model_choice  = request.form.get("model",             "RF")
        scaler_choice = request.form.get("scaler",            "") or None
        fs_tech       = request.form.get("feature_selection", "") or None
        fs_ratio     = int(request.form.get("fs_ratio", 80)) / 100.0  # V3 expects 0.0-1.0
        data_balance = request.form.get("data_balance", "") or None

        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            tmp.write(train_file.read())
            train_path = tmp.name

        test_path = None
        if test_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                tmp.write(test_file.read())
                test_path = tmp.name

        print(f"[ML] V3 | {model_choice} scaler={scaler_choice} fs={fs_tech}({fs_ratio}) balance={data_balance}", flush=True)

        metrics, plt_obj, objects = MachineLearningClassificationV3(
            train_path, scaler_choice, model_choice, fs_tech, fs_ratio,
            dataBalanceTech=data_balance,
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

        # Store _target_size before serialising so the downloaded pkl contains it
        objects["_target_size"] = int(request.form.get("target_size_used", 128))
        pkl_b64 = base64.b64encode(pickle.dumps(objects)).decode()
        key     = str(uuid.uuid4())
        SESSION[key] = {"objects": objects, "type": "ml_model"}

        scalar_metrics = {}
        for k, v in metrics.items():
            try:
                if np.isscalar(v) and not isinstance(v, (str, bool)):
                    scalar_metrics[k] = float(v)
            except Exception:
                pass
        print(f"[ML] Metrics returned: {list(scalar_metrics.keys())}", flush=True)

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
#  TAB 2 — GRID SEARCH (all models × all scalers)
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/grid_search_ml", methods=["POST"])
def grid_search_ml():
    ALL_MODELS  = ["MLP","RF","AB","KNN","DT","ETs","SGD","SVC","GNB","LR","GB"]
    ALL_SCALERS = ["Standard","MinMax","Robust","Normalizer","MaxAbs","QT"]

    train_file = request.files.get("train_csv")
    if not train_file:
        return jsonify({"error": "No training CSV uploaded."}), 400

    test_file    = request.files.get("test_csv")
    target_col   = request.form.get("target_column",     "Class")
    drop_first   = request.form.get("drop_first",        "true").lower() == "true"
    test_ratio   = float(request.form.get("test_ratio",  0.2))
    fs_tech      = request.form.get("feature_selection", "") or None
    fs_ratio     = int(request.form.get("fs_ratio",      80)) / 100.0
    data_balance = request.form.get("data_balance",      "") or None

    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        tmp.write(train_file.read())
        train_path = tmp.name

    test_path = None
    if test_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            tmp.write(test_file.read())
            test_path = tmp.name

    rank_metric  = request.form.get("rank_metric", "Weighted Average").strip()
    total_combos = len(ALL_MODELS) * len(ALL_SCALERS)
    print(f"[GS] Starting grid search: {len(ALL_MODELS)} models x {len(ALL_SCALERS)} scalers = {total_combos} combos | rank_by={rank_metric}", flush=True)

    def _stream():
        history      = []
        done_count   = 0
        best_score   = -1.0
        best_metrics = best_plt = best_objects = None
        best_model_name = best_scaler_name = ""

        for model_name in ALL_MODELS:
            for scaler_name in ALL_SCALERS:
                try:
                    m, p, o = MachineLearningClassificationV3(
                        train_path, scaler_name, model_name, fs_tech, fs_ratio,
                        dataBalanceTech=data_balance,
                        testRatio=test_ratio, testFilePath=test_path,
                        targetColumn=target_col, dropFirstColumn=drop_first,
                    )
                    # Extract all scalar metrics first
                    scalar_m = {}
                    for k, v in m.items():
                        try:
                            if np.isscalar(v) and not isinstance(v, (str, bool)):
                                scalar_m[k] = float(v)
                        except Exception:
                            pass

                    # Ranking criterion: chosen by user, fallback chain if key missing
                    fallbacks = ["Weighted Average","Weighted F1","Macro F1","Macro Accuracy"]
                    score = scalar_m.get(rank_metric, None)
                    if score is None:
                        for fb in fallbacks:
                            if fb in scalar_m:
                                score = scalar_m[fb]; break
                    score = float(score or 0.0)

                    history.append({"Model": model_name, "Scaler": scaler_name,
                                    "Weighted Avg": score,
                                    "Macro F1":     scalar_m.get("Macro F1", 0),
                                    "Macro Accuracy": scalar_m.get("Macro Accuracy", 0)})

                    if score > best_score:
                        best_score = score
                        best_metrics  = scalar_m   # exact same dict used for response
                        best_plt      = p
                        best_objects  = o
                        best_model_name, best_scaler_name = model_name, scaler_name

                    print(f"[GS] {model_name}+{scaler_name}: score={score:.4f} (best={best_score:.4f})", flush=True)

                except Exception as ex:
                    print(f"[GS] {model_name}+{scaler_name} failed: {ex}", flush=True)
                    history.append({"Model": model_name, "Scaler": scaler_name,
                                    "Weighted Avg": 0, "Macro F1": 0, "Macro Accuracy": 0})

                done_count += 1
                pct = round(done_count / total_combos * 100, 1)
                yield f"GRID_PROGRESS:{done_count}/{total_combos}:{pct}:{model_name}:{scaler_name}:{best_score}\n"

        # Sort leaderboard
        history.sort(key=lambda x: x.get("Weighted Avg", 0), reverse=True)

        if not best_objects:
            yield "RESULTS_JSON:" + json.dumps({"error": "All combinations failed."}) + "\n"
            return

        # Store _target_size in best objects
        _ts = request.form.get("target_size_used", "").strip()
        best_objects["_target_size"] = int(_ts) if _ts.isdigit() else 128

        key = str(uuid.uuid4())
        SESSION[key] = {"objects": best_objects, "type": "ml_model"}

        pkl_b64 = base64.b64encode(pickle.dumps(best_objects)).decode()
        cm_b64  = None
        if best_plt is not None:
            try:
                cm_b64 = _fig_to_b64(best_plt)
                plt.close("all")
            except Exception:
                pass

        resp = {
            "best_model":          best_model_name,
            "best_scaler":         best_scaler_name,
            "best_score":          best_score,
            "rank_metric":         rank_metric,
            "metrics":             best_metrics,
            "leaderboard":         history,
            "pkl_b64":             pkl_b64,
            "confusion_matrix_b64":cm_b64,
            "session_key":         key,
        }
        print(f"[GS] Done. Best: {best_model_name}+{best_scaler_name} = {best_score:.4f}", flush=True)
        yield "RESULTS_JSON:" + json.dumps(resp) + "\n"

        # Cleanup temp files
        for p_path in [train_path, test_path]:
            try:
                if p_path: os.unlink(p_path)
            except Exception:
                pass

    return Response(stream_with_context(_stream()), mimetype="text/plain",
                    headers={"X-Accel-Buffering":"no","Cache-Control":"no-cache"})


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
        optimizer_name = body.get("optimizer",   "Adam")
        test_split    = float(body.get("test_split",  0.20))
        val_split     = float(body.get("val_split",   0.25))

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
from tensorflow.keras.optimizers import Adam,SGD,RMSprop,Adamax,Nadam,Adagrad
from tensorflow.keras.preprocessing.image import ImageDataGenerator

print("TF:", tf.__version__, "GPUs:", len(tf.config.list_physical_devices("GPU")))

dataset_dir,output_dir,base_model_name = r"{dataset_dir}",r"{output_dir}","{base_model}"
input_shape = ({img_h},{img_w},3)
batch_size,epochs,patience,lr = {batch_size},{epochs},{patience},{lr}
optimizer_name = "{optimizer_name}"; test_split={test_split}; val_split={val_split}

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

train_df,test_df=train_test_split(df,test_size=test_split,random_state=42,stratify=df["label"])
train_df,val_df =train_test_split(train_df,test_size=val_split,random_state=42,stratify=train_df["label"])
print(f"Split -> train:{{len(train_df)}} val:{{len(val_df)}} test:{{len(test_df)}}")

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
_opt_map={{"Adam":Adam,"SGD":SGD,"RMSprop":RMSprop,"Adamax":Adamax,"Nadam":Nadam,"Adagrad":Adagrad}}
_opt_cls=_opt_map.get(optimizer_name,Adam)
_opt=_opt_cls(lr) if optimizer_name!="SGD" else SGD(lr,momentum=0.9)
model.compile(optimizer=_opt,loss="categorical_crossentropy",metrics=["accuracy"])
print(f"Optimizer: {{optimizer_name}}(lr={{lr}})")

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

        # Always go through ExtractMultipleObjectsFromROI — same as training pipeline
        # Use same target_size as training (stored in pkl, default 128)
        _ts_override = request.form.get("target_size","").strip()
        if _ts_override.isdigit():
            pred_target_size = int(_ts_override)
        else:
            pred_target_size = objects.get("_target_size", None)
            if pred_target_size is None:
                pred_target_size = min(min(img.shape[:2]), 512)
                print(f"[PRED] _target_size missing — inferred {pred_target_size}", flush=True)
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
                # No contours found — fall back to plain resize
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
            probs      = np.array([1.0])   # fallback so SHAP pred_class_idx is safe

        print(f"[PRED] Final result: {pred_label}", flush=True)

        # ── SHAP explanation ───────────────────────────────────────────────────
        shap_b64 = None
        want_shap = request.form.get("shap", "false").lower() == "true"
        if want_shap:
            try:
                import shap as _shap

                # ── Feature names — always match X width exactly ───────────────
                n_feats = X.shape[1]
                if cols is not None and len(cols) > 0:
                    if fs is not None:
                        try:
                            support = fs.get_support()
                            feat_names = [c for c, s in zip(cols, support) if s]
                        except Exception:
                            feat_names = list(cols)
                    else:
                        feat_names = list(cols)
                else:
                    feat_names = [f"f{i}" for i in range(n_feats)]
                # Pad / trim to match X exactly
                if len(feat_names) < n_feats:
                    feat_names += [f"f{i}" for i in range(len(feat_names), n_feats)]
                feat_names = feat_names[:n_feats]

                # ── Predicted class index ──────────────────────────────────────
                pred_class_idx = int(np.argmax(probs))
                n_classes      = len(le.classes_)

                X_explain  = X.astype(np.float64)          # (1, n_feats)
                background = np.zeros((1, n_feats))         # neutral baseline

                # ── Compute raw SHAP values ────────────────────────────────────
                model_type = type(model).__name__
                TREE_MODELS = {"RandomForestClassifier","GradientBoostingClassifier",
                               "ExtraTreesClassifier","AdaBoostClassifier",
                               "DecisionTreeClassifier","XGBClassifier","LGBMClassifier"}

                if model_type in TREE_MODELS:
                    explainer   = _shap.TreeExplainer(model)
                    raw         = explainer.shap_values(X_explain)
                else:
                    # Works for LR, MLP, KNN, SVC, GNB, SGD — everything else
                    predict_fn  = (model.predict_proba
                                   if hasattr(model, "predict_proba")
                                   else model.predict)
                    explainer   = _shap.KernelExplainer(predict_fn, background)
                    raw         = explainer.shap_values(X_explain, nsamples=128,
                                                        silent=True)

                # ── Normalise to 1-D array for predicted class ─────────────────
                # raw can be:
                #   list of (1, n_feats) arrays  — one per class  [common]
                #   ndarray (n_classes, 1, n_feats)               [newer tree]
                #   ndarray (1, n_feats)                          [binary / kernel single]
                #   ndarray (1, n_feats, n_classes)               [unified API]
                if isinstance(raw, list):
                    # list[class_idx] → (1, n_feats) or (n_feats,)
                    idx = min(pred_class_idx, len(raw) - 1)
                    sv  = np.array(raw[idx]).ravel()
                else:
                    arr = np.array(raw)
                    if arr.ndim == 3:
                        if arr.shape[0] == 1:
                            # (1, n_feats, n_classes) — unified API
                            idx = min(pred_class_idx, arr.shape[2] - 1)
                            sv  = arr[0, :, idx]
                        else:
                            # (n_classes, 1, n_feats)
                            idx = min(pred_class_idx, arr.shape[0] - 1)
                            sv  = arr[idx, 0, :]
                    elif arr.ndim == 2:
                        # (1, n_feats) — single output or binary
                        sv = arr[0]
                    else:
                        sv = arr.ravel()

                sv = np.array(sv, dtype=float).ravel()
                # Safety: length must match n_feats
                if len(sv) != n_feats:
                    sv = sv[:n_feats] if len(sv) > n_feats else np.pad(sv, (0, n_feats - len(sv)))

                print(f"[SHAP] {type(explainer).__name__} | class={pred_class_idx} "
                      f"| sv.shape={sv.shape} | "
                      f"top: {feat_names[int(np.argmax(np.abs(sv)))]} "
                      f"({sv[int(np.argmax(np.abs(sv)))]:.4f})", flush=True)

                # ── Bar chart — top 15 features by |SHAP| ─────────────────────
                N     = min(15, n_feats)
                order = np.argsort(np.abs(sv))[::-1][:N]
                vals  = sv[order]                    # highest → lowest importance
                names = [feat_names[i] for i in order]

                fig, ax = plt.subplots(figsize=(7, max(3, N * 0.42)),
                                       facecolor="#0d0d1a")
                ax.set_facecolor("#0d0d1a")
                # Reverse so most important is at top of horizontal bar chart
                bar_colors = ["#ff2d78" if v < 0 else "#00d4ff" for v in vals[::-1]]
                ax.barh(range(N), vals[::-1], color=bar_colors,
                        edgecolor="none", height=0.65)
                ax.set_yticks(range(N))
                ax.set_yticklabels([n[:30] for n in names[::-1]],
                                   fontsize=7, color="#c0c0d0",
                                   fontfamily="monospace")
                ax.set_xlabel("SHAP value  (impact on predicted class score)",
                              fontsize=8, color="#6b7280", labelpad=8)
                ax.set_title(f"SHAP — top features for: {pred_label}",
                             fontsize=9, color="#e8e8f0",
                             fontfamily="monospace", pad=10)
                ax.axvline(0, color="#ffffff33", linewidth=0.8, linestyle="--")
                ax.tick_params(colors="#6b7280", labelsize=7)
                for spine in ax.spines.values():
                    spine.set_edgecolor("#ffffff11")
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                from matplotlib.patches import Patch
                ax.legend(handles=[Patch(color="#00d4ff", label="Pushes toward class"),
                                   Patch(color="#ff2d78", label="Pushes away from class")],
                          fontsize=7, facecolor="#1a1a2e", edgecolor="#ffffff22",
                          labelcolor="#c0c0d0", loc="lower right")
                fig.tight_layout()

                buf = io.BytesIO()
                fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                            facecolor="#0d0d1a")
                plt.close(fig)
                buf.seek(0)
                shap_b64 = base64.b64encode(buf.read()).decode()
                print("[SHAP] Plot encoded OK.", flush=True)

            except Exception as shap_err:
                print(f"[SHAP] Failed: {shap_err}", flush=True)
                traceback.print_exc()
                shap_b64 = None
                shap_err_msg = str(shap_err)

        resp = {"predicted_class": pred_label, "probabilities": probs_dict}
        if shap_b64:
            resp["shap_b64"] = shap_b64
        elif want_shap:
            resp["shap_error"] = locals().get("shap_err_msg", "SHAP did not run")
        return jsonify(resp)

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
            input_hw = int(ci_data.get("input_shape", [input_hw])[0])
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

        # ── Grad-CAM ──────────────────────────────────────────────────────────
        gradcam_b64 = None
        want_gradcam = request.form.get("gradcam", "false").lower() == "true"
        if want_gradcam:
            try:
                # Find the last Conv2D layer
                last_conv = None
                for layer in reversed(cnn_model.layers):
                    if isinstance(layer, tf.keras.layers.Conv2D):
                        last_conv = layer
                        break

                if last_conv is not None:
                    # Build gradient model: inputs -> [last_conv_output, predictions]
                    grad_model = tf.keras.models.Model(
                        inputs  = cnn_model.inputs,
                        outputs = [last_conv.output, cnn_model.output],
                    )
                    img_tensor = tf.cast(img_arr, tf.float32)
                    with tf.GradientTape() as tape:
                        tape.watch(img_tensor)
                        conv_outputs, predictions = grad_model(img_tensor)
                        loss = predictions[:, pred_idx]

                    # Gradients of predicted class w.r.t. conv feature maps
                    grads = tape.gradient(loss, conv_outputs)[0]           # (H, W, C)
                    conv_out = conv_outputs[0]                              # (H, W, C)

                    # Global-average-pool the gradients over spatial dims → weights
                    weights = tf.reduce_mean(grads, axis=(0, 1))           # (C,)

                    # Weighted combination of feature maps
                    cam = tf.reduce_sum(conv_out * weights, axis=-1).numpy()  # (H, W)
                    cam = np.maximum(cam, 0)                                # ReLU
                    if cam.max() > 0:
                        cam = cam / cam.max()

                    # Resize CAM to input image size and colorise
                    cam_resized = cv2.resize(cam, (input_hw, input_hw))
                    heatmap = cv2.applyColorMap(
                        np.uint8(255 * cam_resized), cv2.COLORMAP_JET
                    )                                                       # BGR uint8

                    # Overlay on original RGB image
                    orig_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                    overlay  = cv2.addWeighted(orig_bgr, 0.5, heatmap, 0.5, 0)

                    # Encode to PNG base64
                    _, buf = cv2.imencode(".png", overlay)
                    gradcam_b64 = base64.b64encode(buf.tobytes()).decode()
                    print(f"[GRADCAM] Generated for class {pred_idx} via layer '{last_conv.name}'", flush=True)
                else:
                    print("[GRADCAM] No Conv2D layer found — skipping.", flush=True)
            except Exception as gc_err:
                print(f"[GRADCAM] Failed: {gc_err}", flush=True)
                traceback.print_exc()

        resp = {
            "predicted_class": idx2cls.get(pred_idx, f"Class {pred_idx}"),
            "confidence":      float(preds[pred_idx]) * 100,
            "probabilities":   {idx2cls.get(i, f"Class {i}"): float(preds[i]) for i in range(len(preds))},
        }
        if gradcam_b64:
            resp["gradcam_b64"] = gradcam_b64
        return jsonify(resp)

    except Exception as ex:
        traceback.print_exc()
        return jsonify({"error": str(ex)}), 500


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  NEXUS Flask API v2.3  —  http://localhost:5000")
    print("=" * 55)
    try:
        from waitress import serve
        print("  Using waitress WSGI server (no timeout limits)", flush=True)
        serve(app, host="127.0.0.1", port=5000, threads=4, channel_timeout=3600)
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable,"-m","pip","install","waitress","--quiet"],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        from waitress import serve
        serve(app, host="127.0.0.1", port=5000, threads=4, channel_timeout=3600)
