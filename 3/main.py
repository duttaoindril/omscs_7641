from clustering import main as clustering_main
from dra import main as dra_main
from dra_clustered import main as dra_clustered_main
from nn import main as nn_main
from util import set_seed

if __name__ == "__main__":
    set_seed()
    clustering_main()
    dra_main()
    dra_clustered_main()
    nn_main()