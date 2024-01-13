import taichi as ti
import numpy as np
import time
import scipy
import os

ti.init(arch=ti.cpu)

N = 256
NV = (N + 1) ** 2
NE = 2 * N * (N + 1) + N**2
M = NE
new_M = int(NE / 100)

edge = ti.Vector.field(2, dtype=int, shape=(NE))
pos = ti.Vector.field(3, dtype=float, shape=(NV))
edge_center = ti.Vector.field(3, dtype=ti.float32, shape=(NE))


@ti.kernel
def init_pos(pos: ti.template()):
    for i, j in ti.ndrange(N + 1, N + 1):
        idx = i * (N + 1) + j
        # pos[idx] = ti.Vector([i / N,  j / N, 0.5])  # vertical hang
        pos[idx] = ti.Vector([i / N, 0.5, j / N])  # horizontal hang


@ti.kernel
def init_edge(edge: ti.template()):
    for i, j in ti.ndrange(N + 1, N):
        edge_idx = i * N + j
        pos_idx = i * (N + 1) + j
        edge[edge_idx] = ti.Vector([pos_idx, pos_idx + 1])
    start = N * (N + 1)
    for i, j in ti.ndrange(N, N + 1):
        edge_idx = start + j * N + i
        pos_idx = i * (N + 1) + j
        edge[edge_idx] = ti.Vector([pos_idx, pos_idx + N + 1])
    start = 2 * N * (N + 1)
    for i, j in ti.ndrange(N, N):
        edge_idx = start + i * N + j
        pos_idx = i * (N + 1) + j
        if (i + j) % 2 == 0:
            edge[edge_idx] = ti.Vector([pos_idx, pos_idx + N + 2])
        else:
            edge[edge_idx] = ti.Vector([pos_idx + 1, pos_idx + N + 1])


@ti.kernel
def init_edge_center(
    edge_center: ti.template(),
    edge: ti.template(),
    pos: ti.template(),
):
    for i in range(NE):
        idx1, idx2 = edge[i]
        p1, p2 = pos[idx1], pos[idx2]
        edge_center[i] = (p1 + p2) / 2.0


@ti.kernel
def compute_R_based_on_kmeans_label_triplets(
    labels: ti.types.ndarray(dtype=int),
    ii: ti.types.ndarray(dtype=int),
    jj: ti.types.ndarray(dtype=int),
    vv: ti.types.ndarray(dtype=int),
    new_M: ti.i32,
    M: ti.i32,
):
    cnt = 0
    ti.loop_config(serialize=True)
    for i in range(new_M):
        for j in range(M):
            if labels[j] == i:
                ii[cnt], jj[cnt], vv[cnt] = i, j, 1
                cnt += 1


def compute_R_and_P_kmeans():
    print(">>Computing P and R...")
    t = time.perf_counter()

    from scipy.cluster.vq import vq, kmeans, whiten

    # ----------------------------------- kmans ---------------------------------- #
    print("kmeans start")
    input = edge_center.to_numpy()

    M = NE
    global new_M
    print("M: ", M, "  new_M: ", new_M)

    # run kmeans
    input = whiten(input)
    print("whiten done")

    print("computing kmeans...")
    kmeans_centroids, distortion = kmeans(obs=input, k_or_guess=new_M, iter=1)
    labels, _ = vq(input, kmeans_centroids)

    print("distortion: ", distortion)
    print("kmeans done")
    # ----------------------------------- R and P --------------------------------- #
    # transform labels to R
    i_arr = np.zeros((M), dtype=np.int32)
    j_arr = np.zeros((M), dtype=np.int32)
    v_arr = np.zeros((M), dtype=np.int32)
    compute_R_based_on_kmeans_label_triplets(labels, i_arr, j_arr, v_arr, new_M, M)
    R = scipy.sparse.coo_array((v_arr, (i_arr, j_arr)), shape=(new_M, M)).tocsr()
    P = R.transpose()
    print(f"Computing P and R done, time = {time.perf_counter() - t}")
    return R, P, labels, new_M

print("start computing R and P...")
timer_all = time.perf_counter()
proj_dir_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print("proj_dir_path: ", proj_dir_path)
out_dir = proj_dir_path + f"/data/misc/"
init_pos(pos)
init_edge(edge)
init_edge_center(edge_center, edge, pos)

R, P, labels, new_M = compute_R_and_P_kmeans()
scipy.io.mmwrite(out_dir + "R.mtx", R)
scipy.io.mmwrite(out_dir + "P.mtx", P)
print("R and P saved")
print("time: ", time.perf_counter() - timer_all, " s")
