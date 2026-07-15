from pathlib import Path
import utils

class Dataset:
    def __init__(self, name: str):
        if name not in ["neural_iarap"]:
            raise ValueError(f"Unknown dataset name: {name}")
        
        self.name = name
        self.dataset_dir = Path(__file__).parent / f"data" / "sdf_net_weights"
        self.sdf_paths = [path for path in self.dataset_dir.iterdir() if path.is_file()]

    def __load_sdf(self, path: Path):
        if self.name == "neural_iarap":
            import utils
            return utils.load_neural_sdf(path, dim=3)

    def __len__(self):
        return len(self.sdf_paths)

    def __getitem__(self, idx: int):
        if idx < 0 or idx >= len(self):
            raise IndexError("Index out of range")
        return self.__load_sdf(self.sdf_paths[idx])