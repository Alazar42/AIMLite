"""Client test script for test Custom ML Model."""

from test.data import AppDataset
from test.model import AppModel


def main() -> None:
    print("[*] Initializing test Model...")
    model = AppModel()
    dataset = AppDataset()

    print(f"[*] App dataset ready (file: {getattr(dataset, 'filename', 'dataset.csv')}).")

    # Test sample inference
    sample_inputs = [
        {"feature_1": 1.2, "feature_2": 3.4, "feature_3": 0.5},
        {"feature_1": 2.1, "feature_2": 1.8, "feature_3": 1.4},
    ]

    print("\n[>] Running Client Inference Tests:")
    for sample in sample_inputs:
        output = model.predict(sample)
        print(f"  [Input]  {sample}")
        print(f"  [Output] {output}")


if __name__ == "__main__":
    main()
