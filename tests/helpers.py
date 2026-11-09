"""Shared test helpers."""

from nfds.data.config import config_from_dict


def tiny_config(**overrides):
    d = {
        "name": "tiny", "base_seed": 123, "drag": 0.1,
        "forcing": {"type": "kolmogorov", "wavenumber": 2},
        "simulation": {"sim_n": 32, "train_n": 16, "dt": 0.01, "save_dt": 0.05, "n_frames": 6,
                       "spinup": 0.1, "dtype": "float64", "batch_size": 3},
        "quality": {"max_resolution_ratio": 1.0},
        "splits": {
            "train": {"n_traj": 5, "nu": {"dist": "loguniform", "low": 0.01, "high": 0.05},
                      "ic": {"k_peak": [2, 3]}},
            "val": {"n_traj": 2, "nu": {"dist": "loguniform", "low": 0.01, "high": 0.05}},
            "test_re": {"n_traj": 2, "nu": {"dist": "choice", "values": [0.1, 0.005]},
                        "sim_n": 48, "dt": 0.005},
            "test_ic": {"n_traj": 3, "nu": {"dist": "fixed", "values": [0.02]},
                        "ic": {"type": "mixed"}, "spinup": 0.0, "resolutions": [16, 32]},
        },
    }
    for key, val in overrides.items():
        section, _, field = key.partition("__")
        if field:
            d[section][field] = val
        else:
            d[section] = val
    return config_from_dict(d)
