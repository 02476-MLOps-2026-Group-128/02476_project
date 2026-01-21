from pathlib import Path

from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig

from diabetic_classification.train import run_training
from export_to_onnx.export_to_onnx import export_to_onnx


def test_export_to_onnx(tmp_path) -> None:
    """Test the export_to_onnx function."""
    tmp_models_dir = str(tmp_path / "models")

    with initialize(version_base=None, config_path="../configs/hydra"):
        cfg = compose(
            config_name="config",
            return_hydra_config=True,
            overrides=[
                f"trainer.models_dir={tmp_models_dir}",
                "trainer.max_epochs=1",  # We don't need performance here
            ],
        )
        HydraConfig.instance().set_config(cfg)

        checkpoint_path = run_training(cfg)

        assert checkpoint_path is not None
        assert Path(checkpoint_path).exists()
        assert checkpoint_path.endswith(".ckpt")

    GlobalHydra.instance().clear()
    onnx_output_path = export_to_onnx(checkpoint_path)

    assert onnx_output_path is not None
    assert Path(onnx_output_path).exists()
    assert str(onnx_output_path).endswith(".onnx")
