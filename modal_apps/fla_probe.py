import modal

from modal_apps.train_app import image

app = modal.App("rein-fla-probe")


@app.function(image=image, timeout=600)
def probe():
    import importlib.metadata as md
    import subprocess

    for dist in ("flash-linear-attention", "fla-core", "fla"):
        try:
            print(dist, md.version(dist))
        except md.PackageNotFoundError:
            print(dist, "not found")
    try:
        import fla

        print("import fla ok", getattr(fla, "__version__", "?"), fla.__file__)
    except Exception as e:
        print("import fla failed:", type(e).__name__, e)
    from transformers.utils import import_utils as iu

    print("_is_package_available(fla):", iu._is_package_available("fla", return_version=True))
    print(subprocess.run(["pip", "show", "flash-linear-attention"], capture_output=True, text=True).stdout)
