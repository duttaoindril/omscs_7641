from frozen_lake import main as frozen_lake_main
from forest import main as forest_main
from util import set_seed

if __name__ == "__main__":
    set_seed()
    forest_main()
    frozen_lake_main()