"""Version-compatible loading for known bundled or explicitly trusted weights."""
import hashlib
import io
import pickle
from pathlib import Path

import torch


# Verified during packaging. A matching filename alone never establishes trust.
BUNDLED_CHECKPOINT_SHA256 = {
    "33ec95baddb2a737e478e973ff2421726b7e969a97fb75138ab0ebd9197230ae",
    "97fe05ce8df323a1ec4f62704e5e7f2f5776a94140748e4a1df1a9fe267947d5",
}


def load_checkpoint(path, trusted_checkpoint=False):
    """Prefer restricted loading; old APIs may read only known/trusted bytes.

    Retain one byte snapshot so a file changed between verification and loading
    cannot substitute a different checkpoint for the verified bundle.
    """
    path = Path(path)
    data = path.read_bytes()
    try:
        return torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
    except TypeError as error:
        message = str(error).lower()
        if "weights_only" not in message or not any(s in message for s in ("invalid keyword", "unexpected keyword")):
            raise
        known_bundle = hashlib.sha256(data).hexdigest() in BUNDLED_CHECKPOINT_SHA256
        if not (trusted_checkpoint or known_bundle):
            raise ValueError(
                "This PyTorch has no weights_only support. Use the bundled verified "
                "models, or set trusted_checkpoint=True for your own checkpoint "
                "(trusted_sequence_checkpoint for fitting; finding.trusted_checkpoint "
                "for finding). No unrestricted fallback was performed."
            ) from error
        kind = "verified bundled" if known_bundle else "explicitly trusted"
        print("Checkpoint compatibility: PyTorch %s | legacy API | %s: %s" %
              (torch.__version__, kind, path.name), flush=True)
        return torch.load(io.BytesIO(data), map_location="cpu")
    except pickle.UnpicklingError as error:
        if not trusted_checkpoint:
            raise ValueError(
                "Restricted checkpoint loading failed. For your own original training "
                "checkpoint containing optimizer/NumPy RNG state, explicitly set "
                "trusted_checkpoint=True. No unrestricted fallback was performed. "
                "Original error: " + str(error)
            ) from error
        return torch.load(io.BytesIO(data), map_location="cpu", weights_only=False)
