import pytest
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra


DEFAULT_OVERRIDES = [
    
    "paths.project_root=./",
    "data.data_dir=./data",
]


@pytest.fixture(autouse=True)
def disable_wandb(monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.setenv("WANDB_SILENT", "true")


@pytest.fixture()
def cfg_factory():
    def _factory(extra_overrides=None):
        overrides = list(DEFAULT_OVERRIDES)
        if extra_overrides:
            overrides.extend(extra_overrides)

        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()

        with initialize(version_base=None, config_path="../configs/hydra"):
            return compose(config_name="config", overrides=overrides)

    return _factory


@pytest.fixture()
def cfg(cfg_factory):
    return cfg_factory()

@pytest.fixture(autouse=True)
def finish_wandb(disable_wandb):
    try:
        import wandb
    except Exception:
        yield
        return

    if wandb.run is not None:
        wandb.finish()

    yield

    if wandb.run is not None:
        wandb.finish()
