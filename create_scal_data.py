import numpy as np
import pandas as pd
import os
from tqdm import trange


dataset_root = f'data/scal_data/'

os.makedirs(dataset_root, exist_ok=True)
def generate_data(n_nor, n_ano, dim, n_nor_c=5, n_ano_c=5):
    # normal class with "n_nor_c" clusters
    x_nor = np.zeros([n_nor, dim])
    for i in trange(dim):
        size = round(n_nor / n_nor_c)
        for j in range(n_nor_c):
            loc = np.random.rand()
            scale = float(np.random.rand())
            # print("Inlier: dim" + str(i), "cluster" + str(j), round(loc, 1), round(scale, 2))
            # last c
            if j == n_nor_c - 1:
                last_size = n_nor - (n_nor_c - 1) * size
                x_nor[j * size:, i] = np.random.normal(loc, scale, last_size)
            else:
                x_nor[j * size: (j + 1) * size, i] = np.random.normal(loc, scale, size)

    x_ano = np.zeros([n_ano, dim])
    for i in trange(dim):
        size = round(n_ano / n_ano_c)
        for j in range(n_ano_c):
            loc = np.random.rand() + 1
            scale = float(np.random.rand())
            # print("anomaly: dim" + str(i), "cluster" + str(j), round(loc, 1), round(scale, 2))

            # last c
            if j != n_ano_c - 1:
                x_ano[j * size: (j + 1) * size, i] = np.random.normal(loc, scale, size)
            else:
                last_size = n_ano - (n_ano_c - 1) * size
                x_ano[j * size:, i] = np.random.normal(loc, scale, last_size)
            # x_ano[:, i] = np.random.normal(loc, scale, n_ano)

    x = np.concatenate([x_ano, x_nor], axis=0)
    y = np.append(np.ones(n_ano, dtype=int), np.zeros(n_nor, dtype=int))
    matrix = np.concatenate([x, y.reshape([x.shape[0], 1])], axis=1)

    columns = ["A" + str(i) for i in range(dim)]
    columns.append("class")
    df = pd.DataFrame(matrix, columns=columns)
    df['class'] = df['class'].astype(int)
    return df


# Scal-up Test
dim_range = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
size_range = [1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000, 256000]

for ii, dim in enumerate(dim_range):
    n_nor = 4500
    n_ano = 500
    size = n_nor + n_ano
    df = generate_data(n_nor=n_nor, n_ano=n_ano, dim=dim)
    name = 'scal_dim-' + str(ii) + "@" + str(size) + "-" + str(dim) + ".csv"
    df.to_csv(os.path.join(dataset_root, name), index=False)

for ii, size in enumerate(size_range):
    dim = 32
    n_nor = int(size * 0.95)
    n_ano = int(size * 0.05)
    df = generate_data(n_nor=n_nor, n_ano=n_ano, dim=dim)
    name = "scal_size-" + str(ii) + "@" + str(size) + "-" + str(dim) + ".csv"
    df.to_csv(os.path.join(dataset_root, name), index=False)
