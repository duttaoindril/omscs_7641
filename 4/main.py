from frozen_lake import main as frozen_lake_main
from forest import main as forest_main
from taxi import main as taxi_main
from util import set_seed

if __name__ == "__main__":
    frozen_lake_main()
    forest_main()
    taxi_main()
    set_seed()