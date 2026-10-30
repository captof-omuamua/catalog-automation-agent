"""Write the product, gold and QC files to data/ (dev 2,000 / test 1,610 / load 5,000)."""

from catalog_agent.config import settings
from catalog_agent.data import build

if __name__ == "__main__":
    print(build(settings.iave_root, settings.data_dir))
