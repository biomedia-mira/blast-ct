"""On-demand download of model weights and large atlas files.

Heavy assets (trained model checkpoints and NIfTI templates) are not shipped
inside the PyPI wheel. They are fetched from the GitHub repository the first
time they are needed and cached under the user's cache directory.
"""
import hashlib
import os
import shutil
import sys
import tempfile
import urllib.request

# Pinned to the commit that contains the assets. Update if assets move.
ASSETS_REF = os.environ.get(
    "BLAST_CT_ASSETS_REF", "ccfa6efb6ee269b97d971a96658d78953381c08e"
)
ASSETS_BASE_URL = (
    f"https://raw.githubusercontent.com/biomedia-mira/blast-ct/{ASSETS_REF}/blast_ct/data"
)

NUM_MODELS = 15

# Relative paths (inside data/) of assets fetched on demand.
REMOTE_ASSETS = [f"saved_models/model_{i}.torch_model" for i in range(1, NUM_MODELS + 1)] + [
    "localisation_files/ct_template.nii.gz",
    "localisation_files/atlas_template_space.nii.gz",
    "localisation_files/ct_template_mask.nii.gz",
]


def _cache_root():
    override = os.environ.get("BLAST_CT_CACHE_DIR")
    if override:
        return override
    if sys.platform == "darwin":
        return os.path.expanduser("~/Library/Caches/blast_ct")
    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg:
        return os.path.join(xdg, "blast_ct")
    return os.path.expanduser("~/.cache/blast_ct")


def _package_data_dir():
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), "data")


def _download(url, dest):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(f"[blast_ct] Downloading {url}", file=sys.stderr)
    with tempfile.NamedTemporaryFile(delete=False, dir=os.path.dirname(dest)) as tmp:
        tmp_path = tmp.name
    try:
        with urllib.request.urlopen(url) as response, open(tmp_path, "wb") as out:
            shutil.copyfileobj(response, out)
        os.replace(tmp_path, dest)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def get_asset_path(relpath):
    """Return a filesystem path to a data asset, downloading it if needed.

    Looks first inside the installed package's ``data/`` directory (for assets
    that are bundled, e.g. during development), then in the user cache. Missing
    remote assets are downloaded into the cache.
    """
    bundled = os.path.join(_package_data_dir(), relpath)
    if os.path.exists(bundled):
        return bundled

    cached = os.path.join(_cache_root(), relpath)
    if os.path.exists(cached):
        return cached

    if relpath not in REMOTE_ASSETS:
        raise FileNotFoundError(
            f"Asset {relpath!r} is not bundled and not registered for download."
        )

    url = f"{ASSETS_BASE_URL}/{relpath}"
    _download(url, cached)
    return cached


def get_model_path(index):
    return get_asset_path(f"saved_models/model_{index}.torch_model")


def get_model_paths(count):
    return [get_model_path(i) for i in range(1, count + 1)]


def get_localisation_dir():
    """Ensure the localisation NIfTI files are available and return their dir."""
    for name in (
        "ct_template.nii.gz",
        "atlas_template_space.nii.gz",
        "ct_template_mask.nii.gz",
    ):
        get_asset_path(f"localisation_files/{name}")
    # Small files (atlas_labels.csv) are bundled; large ones may live in cache.
    # Return the dir containing the large templates.
    bundled = os.path.join(_package_data_dir(), "localisation_files")
    cached = os.path.join(_cache_root(), "localisation_files")
    # Copy the small bundled csv next to the cached templates so callers see
    # a single directory containing every localisation file they need.
    labels_src = os.path.join(bundled, "atlas_labels.csv")
    if os.path.exists(os.path.join(cached, "ct_template.nii.gz")):
        labels_dst = os.path.join(cached, "atlas_labels.csv")
        if os.path.exists(labels_src) and not os.path.exists(labels_dst):
            os.makedirs(cached, exist_ok=True)
            shutil.copyfile(labels_src, labels_dst)
        return cached
    return bundled


def get_config_path():
    return os.path.join(_package_data_dir(), "config.json")


def prefetch_all():
    """Download every remote asset up front. Useful for offline setups."""
    for rel in REMOTE_ASSETS:
        get_asset_path(rel)
