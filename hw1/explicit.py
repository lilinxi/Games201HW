from framwork import *


@ti.kernel
def substep():
    # Compute force and new velocity
    n = num_particles[None]
    for i in range(n):
        total_force = ti.Vector(gravity) * particle_mass
        for j in range(n):
            if rest_length[i, j] != 0:
                x_ij = x[i] - x[j]
                v_ij = v[i] - v[j]
                total_force += -damping[None] * x_ij.normalized() * v_ij * x_ij.normalized()  # damping
                total_force += -spring_stiffness[None] * (x_ij.norm() - rest_length[i, j]) * x_ij.normalized()  # spring
        v[i] += dt * total_force / particle_mass


init_mass_spring_system()

while True:
    process_input()

    if not paused[None]:
        for step in range(10):
            substep()
            collide_with_ground()
            update_position()
        compute_current_energy()
    process_output()
