"""XPCS data loading from HDF5, NPZ, and MAT files."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import h5py
import numpy as np

from heterodyne.utils.logging import get_logger
from heterodyne.utils.path_validation import validate_file_exists

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)


@dataclass
class XPCSData:
    """Container for loaded XPCS data."""
    
    # Two-time correlation matrix c2(t1, t2)
    c2: np.ndarray
    
    # Time arrays
    t1: np.ndarray
    t2: np.ndarray
    
    # Optional metadata
    q: float | None = None
    phi_angles: np.ndarray | None = None
    uncertainties: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    
    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of correlation data."""
        return self.c2.shape
    
    @property
    def n_times(self) -> int:
        """Number of time points (assumes square c2)."""
        return self.c2.shape[0]
    
    @property
    def has_multi_phi(self) -> bool:
        """Whether data has multiple phi angles."""
        return self.c2.ndim == 3


class XPCSDataLoader:
    """Loader for XPCS correlation data from various file formats."""
    
    def __init__(
        self,
        file_path: Path | str,
        format: str | None = None,
    ) -> None:
        """Initialize loader.
        
        Args:
            file_path: Path to data file
            format: File format ('hdf5', 'npz', 'mat'), or None to auto-detect
        """
        self.file_path = validate_file_exists(file_path, "XPCS data file")
        self.format = format or self._detect_format()
    
    def _detect_format(self) -> str:
        """Detect file format from extension."""
        suffix = self.file_path.suffix.lower()
        format_map = {
            ".h5": "hdf5",
            ".hdf5": "hdf5",
            ".hdf": "hdf5",
            ".npz": "npz",
            ".npy": "npy",
            ".mat": "mat",
        }
        if suffix not in format_map:
            raise ValueError(f"Unknown file format: {suffix}")
        return format_map[suffix]
    
    def load(
        self,
        c2_key: str = "c2",
        time_key: str = "t",
        q_key: str | None = "q",
        phi_key: str | None = "phi",
    ) -> XPCSData:
        """Load XPCS data from file.
        
        Args:
            c2_key: Key/path for correlation data
            time_key: Key/path for time array
            q_key: Optional key for wavevector
            phi_key: Optional key for phi angles
            
        Returns:
            XPCSData container
        """
        logger.info(f"Loading XPCS data from {self.file_path}")
        
        if self.format == "hdf5":
            return self._load_hdf5(c2_key, time_key, q_key, phi_key)
        elif self.format == "npz":
            return self._load_npz(c2_key, time_key, q_key, phi_key)
        elif self.format == "npy":
            return self._load_npy()
        elif self.format == "mat":
            return self._load_mat(c2_key, time_key, q_key, phi_key)
        else:
            raise ValueError(f"Unsupported format: {self.format}")
    
    def _load_hdf5(
        self,
        c2_key: str,
        time_key: str,
        q_key: str | None,
        phi_key: str | None,
    ) -> XPCSData:
        """Load from HDF5 file."""
        with h5py.File(self.file_path, "r") as f:
            # Load correlation data
            if c2_key not in f:
                available = list(f.keys())
                raise KeyError(f"Key '{c2_key}' not found. Available: {available}")
            c2 = np.asarray(f[c2_key], dtype=np.float64)

            # Load time array
            if time_key in f:
                t = np.asarray(f[time_key], dtype=np.float64)
            else:
                logger.warning(f"Time key '{time_key}' not found, using indices")
                t = np.arange(c2.shape[0], dtype=np.float64)
            
            # Load optional q
            q = float(f[q_key][()]) if q_key and q_key in f else None
            
            # Load optional phi angles
            phi = None
            if phi_key and phi_key in f:
                phi = np.asarray(f[phi_key], dtype=np.float64)
            
            # Collect metadata (sanitize HDF5 attribute types)
            metadata = {}
            for key in f.attrs:
                value = f.attrs[key]
                if isinstance(value, bytes):
                    value = value.decode('utf-8')
                elif isinstance(value, np.generic):
                    value = value.item()
                metadata[key] = value
        
        return XPCSData(
            c2=c2,
            t1=t,
            t2=t,
            q=q,
            phi_angles=phi,
            metadata=metadata,
        )
    
    def _load_npz(
        self,
        c2_key: str,
        time_key: str,
        q_key: str | None,
        phi_key: str | None,
    ) -> XPCSData:
        """Load from NPZ file.

        Security: allow_pickle=False prevents arbitrary code execution
        from untrusted NPZ files.
        """
        try:
            npz_file = np.load(self.file_path, allow_pickle=False)
        except ValueError:
            raise ValueError(
                f"NPZ file {self.file_path} contains pickled objects, "
                "which is not allowed for security"
            )

        with npz_file as data:
            if c2_key not in data:
                available = list(data.keys())
                raise KeyError(f"Key '{c2_key}' not found. Available: {available}")
            c2 = np.asarray(data[c2_key], dtype=np.float64)

            if time_key in data:
                t = np.asarray(data[time_key], dtype=np.float64)
            else:
                t = np.arange(c2.shape[0], dtype=np.float64)

            q = float(data[q_key]) if q_key and q_key in data else None
            phi = None
            if phi_key and phi_key in data:
                phi = np.asarray(data[phi_key], dtype=np.float64)

        return XPCSData(
            c2=c2,
            t1=t,
            t2=t,
            q=q,
            phi_angles=phi,
        )
    
    def _load_npy(self) -> XPCSData:
        """Load from NPY file (just the array)."""
        c2 = np.load(self.file_path)
        t = np.arange(c2.shape[0])
        return XPCSData(c2=c2, t1=t, t2=t)
    
    def _load_mat(
        self,
        c2_key: str,
        time_key: str,
        q_key: str | None,
        phi_key: str | None,
    ) -> XPCSData:
        """Load from MATLAB .mat file."""
        from scipy.io import loadmat
        
        data = loadmat(self.file_path)
        
        if c2_key not in data:
            # Filter out MATLAB internal keys
            available = [k for k in data.keys() if not k.startswith("__")]
            raise KeyError(f"Key '{c2_key}' not found. Available: {available}")
        
        c2 = np.asarray(data[c2_key], dtype=np.float64)

        if time_key in data:
            t = np.asarray(data[time_key], dtype=np.float64).ravel()
        else:
            t = np.arange(c2.shape[0], dtype=np.float64)

        q = float(data[q_key].ravel()[0]) if q_key and q_key in data else None
        phi = None
        if phi_key and phi_key in data:
            phi = np.asarray(data[phi_key], dtype=np.float64).ravel()
        
        return XPCSData(
            c2=c2,
            t1=t,
            t2=t,
            q=q,
            phi_angles=phi,
        )


def load_xpcs_data(
    file_path: Path | str,
    c2_key: str = "c2",
    time_key: str = "t",
    format: str | None = None,
) -> XPCSData:
    """Convenience function to load XPCS data.
    
    Args:
        file_path: Path to data file
        c2_key: Key for correlation data
        time_key: Key for time array
        format: File format (auto-detected if None)
        
    Returns:
        XPCSData container
    """
    loader = XPCSDataLoader(file_path, format=format)
    return loader.load(c2_key=c2_key, time_key=time_key)
