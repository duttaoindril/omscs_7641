import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.random_projection import GaussianRandomProjection
from sklearn.metrics.pairwise import euclidean_distances as euclid_dist
from sklearn.decomposition import PCA, FastICA, TruncatedSVD
from data import yoga_count, get_yoga_data, get_stock_data, stock_count
from util import set_seed, start_time, kurtosis, plt, plot


def pca(data, count):
    end = start_time('PCA')
    pca = PCA(n_components=count).fit(data)
    transformed_data = pca.transform(data)
    kurt = kurtosis(transformed_data)
    dist = np.mean(abs(euclid_dist(data) - euclid_dist(transformed_data)))

    reconstructed_data = pca.inverse_transform(transformed_data)
    error = mean_squared_error(data, reconstructed_data)
    time = end()
    return transformed_data, reconstructed_data, error, time, kurt, dist


def ica(data, count):
    end = start_time('ICA')
    pca = FastICA(n_components=count).fit(data)
    transformed_data = pca.transform(data)
    kurt = kurtosis(transformed_data)
    dist = np.mean(abs(euclid_dist(data) - euclid_dist(transformed_data)))

    reconstructed_data = pca.inverse_transform(transformed_data)
    error = mean_squared_error(data, reconstructed_data)
    time = end()
    return transformed_data, reconstructed_data, error, time, kurt, dist


def rca(data, count):
    end = start_time('RCA')
    rca = GaussianRandomProjection(n_components=count).fit(data)
    transformed_data = rca.transform(data)
    kurt = kurtosis(transformed_data)
    dist = np.mean(abs(euclid_dist(data) - euclid_dist(transformed_data)))

    components = rca.components_
    p_inverse = np.linalg.pinv(components.T)
    reconstructed_data = transformed_data.dot(p_inverse)
    assert data.shape == reconstructed_data.shape
    error = mean_squared_error(data, reconstructed_data)
    time = end()
    return transformed_data, reconstructed_data, error, time, kurt, dist


def lsa(data, count):
    end = start_time('LSA')
    lsa = TruncatedSVD(n_components=count).fit(data)
    transformed_data = lsa.transform(data)
    kurt = kurtosis(transformed_data)
    dist = np.mean(abs(euclid_dist(data) - euclid_dist(transformed_data)))

    reconstructed_data = lsa.inverse_transform(transformed_data)
    error = mean_squared_error(data, reconstructed_data)
    time = end()
    return transformed_data, reconstructed_data, error, time, kurt, dist

dra = {
    'pca': pca,
    'ica': ica,
    'rca': rca,
    'lsa': lsa,
}

def run(name, features, labels, feature_count):
    labels = labels.values
    component_counts = list(range(1, feature_count))
    print(name+" DRA PCA")
    pca_error = []
    pca_time = []
    pca_kurt = []
    pca_dist = []
    for i in component_counts:
        _, _, error, time, kurt, dist = pca(features, i)
        print(i, error, time, kurt, dist)
        pca_error.append(error)
        pca_time.append(time)
        pca_kurt.append(kurt)
        pca_dist.append(dist)
    print(name+" DRA ICA")
    ica_error = []
    ica_time = []
    ica_kurt = []
    ica_dist = []
    for i in component_counts:
        _, _, error, time, kurt, dist = ica(features, i)
        print(i, error, time, kurt, dist)
        ica_error.append(error)
        ica_time.append(time)
        ica_kurt.append(kurt)
        ica_dist.append(dist)
    print(name+" DRA RCA")
    rca_error = []
    rca_time = []
    rca_kurt = []
    rca_dist = []
    for i in component_counts:
        _, _, error, time, kurt, dist = rca(features, i)
        print(i, error, time, kurt, dist)
        rca_error.append(error)
        rca_time.append(time)
        rca_kurt.append(kurt)
        rca_dist.append(dist)
    print(name+" DRA LSA")
    lsa_error = []
    lsa_time = []
    lsa_kurt = []
    lsa_dist = []
    for i in component_counts:
        _, _, error, time, kurt, dist = lsa(features, i)
        print(i, error, time, kurt, dist)
        lsa_error.append(error)
        lsa_time.append(time)
        lsa_kurt.append(kurt)
        lsa_dist.append(dist)
    plt(component_counts, pca_error, label="PCA")
    plt(component_counts, ica_error, label="ICA")
    plt(component_counts, rca_error, label="RCA")
    plt(component_counts, lsa_error, label="LSA")
    plot(
        name+' PCA ICA RCA LSA Error per Component Count',
        xlabel="Component Count",
        ylabel="Error",
    )
    plt(component_counts, pca_time, label="PCA")
    plt(component_counts, ica_time, label="ICA")
    plt(component_counts, rca_time, label="RCA")
    plt(component_counts, lsa_time, label="LSA")
    plot(
        name+' PCA ICA RCA LSA Time per Component Count',
        xlabel="Component Count",
        ylabel="Time",
    )
    plt(component_counts, pca_kurt, label="PCA")
    plt(component_counts, ica_kurt, label="ICA")
    plt(component_counts, rca_kurt, label="RCA")
    plt(component_counts, lsa_kurt, label="LSA")
    plot(
        name+' PCA ICA RCA LSA Kurtosis per Component Count',
        xlabel="Component Count",
        ylabel="Kurtosis",
    )
    plt(component_counts, pca_dist, label="PCA")
    plt(component_counts, ica_dist, label="ICA")
    plt(component_counts, rca_dist, label="RCA")
    plt(component_counts, lsa_dist, label="LSA")
    plot(
        name+' PCA ICA RCA LSA Distance per Component Count',
        xlabel="Component Count",
        ylabel="Distance",
    )


def yoga():
    features, labels, _, _ = get_yoga_data(test_split=0)
    run('Yoga', features, labels, yoga_count)


def stock(norm=False):
    norm_string = "Normalized " if norm else ""
    features, labels, _, _ = get_stock_data(
        test_split=0,
        normalized=norm,
    )
    run(norm_string + 'Stock', features, labels, stock_count)


def main():
    yoga()
    stock()
    stock(norm=True)


if __name__ == "__main__":
    set_seed()
    main()
