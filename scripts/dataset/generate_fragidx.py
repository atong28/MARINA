from src.modules.data.fp_loader import EntropyFPLoader
from typing import Dict

def build_retrieval_data() -> Dict[str, Dict]:
    fp_loader = EntropyFPLoader(retrieval_path='data/dataset/retrieval.pkl')
    fp_loader.setup(16384, 6)
    return

if __name__ == "__main__":
    build_retrieval_data()