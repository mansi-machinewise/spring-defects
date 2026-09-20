"""Train PatchCore with good Camera 1 images only, stored in data/raw/good/."""
import argparse

from anomalib.data import Folder
from anomalib.engine import Engine
from anomalib.models import Patchcore


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--good-dir", default="data/raw/good")
    args = parser.parse_args()
    datamodule = Folder(
        name="spring",
        root=args.good_dir,
        normal_dir=".",
        train_batch_size=2,
        eval_batch_size=2,
        num_workers=0,
    )
    engine = Engine(default_root_dir="models/patchcore")
    engine.fit(model=Patchcore(backbone="wide_resnet50_2"), datamodule=datamodule)
    print("Training complete. Set patchcore.checkpoint_path to the generated best checkpoint.")


if __name__ == "__main__":
    main()
