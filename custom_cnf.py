"""
Example custom configuration entry point for training.

Edit the `config = TrainingConfig(...)` block to point to your model path,
data.yaml, output directory, and any dataset caching options. Then run:

    python custom_cnf.py
"""

from config import TrainingConfig, LoRAConfig
from train import main


def build_config() -> TrainingConfig:
    """
    Create a custom TrainingConfig. Update the values below as needed.
    """
    return TrainingConfig(
        # Paths
        model_path="/path/to/models/florence-2-large",
        data_yaml_path="/path/to/data.yaml",
        output_dir="./outputs/custom_run",
        # Common training tweaks
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=1e-4,
        num_train_epochs=3,
        # Dataset preprocessing
        num_preprocessing_workers=8,
        save_visualizations=True,
        num_visualization_samples=10,
        # Optional: save or load converted Florence-format datasets to skip reprocessing
        save_converted_dataset=True,
        load_converted_dataset=False,
        converted_dataset_dir="./converted_florence_dataset",
        # LoRA settings (adjust if needed)
        # You can tune r / alpha / dropout and target_modules if desired.
        lora=LoRAConfig(
            r=32,
            lora_alpha=64,
            lora_dropout=0.1,
            # target_modules defaults to ["q_proj", "k_proj", "v_proj", "out_proj"]
            # target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        ),
        # Mixed precision
        fp16=True,
        bf16=False,
    )


if __name__ == "__main__":
    cfg = build_config()
    main(cfg)


