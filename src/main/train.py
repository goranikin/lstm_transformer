import hydra
from omegaconf import DictConfig

from src.main.pipelines.training import run_pipeline


@hydra.main(
    version_base=None, config_path="../../configs", config_name="train_am_tsp_rl"
)
def main(cfg: DictConfig) -> None:
    run_pipeline(cfg)


if __name__ == "__main__":
    main()
