"""Build the eight-class Log-Mel dataset with the shared comparison protocol."""

from pathlib import Path

from data_sugment_MFCC import main


if __name__ == "__main__":
    main(
        default_feature="logmel",
        default_output_dir=Path("dataset_processing")
        / "output"
        / "LogMel_dataset_A_8class",
    )
