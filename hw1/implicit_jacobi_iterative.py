from implicit_framwork import *


# D = diag(A), E = A - D
# Ax = b
# Dx = -Ex + b
# x(i+1) = -D^-1Ex(i) + D-1b
@ti.kernel
def jacobi_iteration():
    n = num_particles[None]
    for i in range(n):
        r = b[i]
        for j in range(n):
            if i != j:
                r -= A[i, j] @ delta_v[j]

        new_delta_v[i] = A[i, i].inverse() @ r

    for i in range(n):
        delta_v[i] = new_delta_v[i]


@ti.kernel
def jacobi_residual() -> ti.f32:
    n = num_particles[None]
    res = 0.0
    for i in range(n):
        r = b[i] * 1.0
        for j in range(n):
            r -= A[i, j] @ delta_v[j]
        res += r.norm()

    return res


def substep(iter_times=10):
    update_mass_matrix()
    update_jacobi_matrix()
    update_A_matrix()
    update_F_vector()
    update_b_vector()

    for step in range(iter_times):
        jacobi_iteration()
        # print(f'iter {step}, residual={jacobi_residual():0.10f}')

    update_velocity()
    update_position()
    collide_with_ground()


init_mass_spring_system()

while True:
    process_input()

    if not paused[None]:
        for step in range(10):
            substep(10)
            compute_damp_energy()
        compute_current_energy()

    process_output()
