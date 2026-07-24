"""Build the eight-class PCEN dataset with the shared comparison protocol."""

from pathlib import Path

from data_sugment_MFCC import main


if __name__ == "__main__":
    main(
        default_feature="pcen",
        default_output_dir=Path("dataset_processing")
        / "output"
        / "PCEN_dataset_A_8class",
    )
