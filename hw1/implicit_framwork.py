from framwork import *

# mass matrix of diag(m1, m2, ..., mn)
M = ti.Matrix(2, 2, dt=ti.f32, shape=(max_num_particles, max_num_particles))
# jacobi matrix of partial_f / partial_x
Jx = ti.Matrix(2, 2, dt=ti.f32, shape=(max_num_particles, max_num_particles))
# jacobi matrix of partial_f / partial_v
Jv = ti.Matrix(2, 2, dt=ti.f32, shape=(max_num_particles, max_num_particles))
# A = [M - dt^2 * Jx - dt * Jv]
A = ti.Matrix(2, 2, dt=ti.f32, shape=(max_num_particles, max_num_particles))
# force vector
F = ti.Vector(2, dt=ti.f32, shape=max_num_particles)
# b = dt * F + dt * dt * Jx * vt
b = ti.Vector(2, dt=ti.f32, shape=max_num_particles)

# iteration temp variables
delta_v = ti.Vector(2, dt=ti.f32, shape=max_num_particles)
new_delta_v = ti.Vector(2, dt=ti.f32, shape=max_num_particles)


@ti.kernel
def update_mass_matrix():
    m = ti.Matrix([
        [particle_mass, 0],
        [0, particle_mass]
    ])
    for i in range(num_particles[None]):
        M[i, i] = m


@ti.kernel
def update_jacobi_matrix():
    I = ti.Matrix([
        [1.0, 0.0],
        [0.0, 1.0]
    ])
    for i, d in Jx:
        Jx[i, d] *= 0.0  # 矩阵清零
        Jv[i, d] *= 0.0  # 矩阵清零
        for j in range(num_particles[None]):
            l_ij = rest_length[i, j]
            if (l_ij != 0) and (d == i or d == j):
                x_ij = x[i] - x[j]
                x_ij_norm = x_ij.norm()
                x_ij_normalized = x_ij / x_ij_norm
                x_ij_mat = x_ij_normalized.outer_product(x_ij_normalized)  # 张量积

                jx = -spring_stiffness[None] * ((1 - l_ij / x_ij_norm) * (I - x_ij_mat) + x_ij_mat)
                if d == i:
                    Jx[i, d] += jx
                elif d == j:
                    Jx[i, d] += -jx

                jv = -damping[None] * x_ij_mat
                if d == i:
                    Jv[i, d] += jv
                elif d == j:
                    Jv[i, d] += -jv


@ti.kernel
def update_A_matrix():
    for i, j in A:
        A[i, j] = M[i, j] - dt ** 2 * Jx[i, j] - dt * Jv[i, j]


@ti.kernel
def update_F_vector():
    n = num_particles[None]
    for i in range(n):
        F[i] = ti.Vector(gravity) * particle_mass
        for j in range(n):
            l_ij = rest_length[i, j]
            if l_ij != 0:
                x_ij = x[i] - x[j]
                v_ij = v[i] - v[j]
                x_ij_norm = x_ij.norm()
                x_ij_normalized = x_ij / x_ij_norm
                F[i] += -spring_stiffness[None] * (x_ij_norm - l_ij) * x_ij_normalized  # spring
                F[i] += -damping[None] * x_ij_normalized * v_ij * x_ij_normalized  # damping


@ti.kernel
def update_b_vector():
    n = num_particles[None]
    for i in range(n):
        b[i] = dt * F[i]
        for j in range(n):
            b[i] += dt ** 2 * Jx[i, j] @ v[j]  # *：元素积，@：矩阵积


@ti.kernel
def update_velocity():
    for i in range(num_particles[None]):
        v[i] += delta_v[i]
