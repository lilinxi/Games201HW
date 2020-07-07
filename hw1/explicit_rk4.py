from framwork import *


@ti.kernel
def substep():
    # Compute force and new velocity
    n = num_particles[None]
    for i in range(n):
        v1 = v[i]
        total_force = ti.Vector(gravity) * particle_mass
        for j in range(n):
            if rest_length[i, j] != 0:
                x_ij = x[i] - x[j]
                v_ij = v[i] - v[j]
                total_force += -damping[None] * x_ij.normalized() * v_ij * x_ij.normalized()  # damping
                total_force += -spring_stiffness[None] * (x_ij.norm() - rest_length[i, j]) * x_ij.normalized()  # spring
        a1 = total_force / particle_mass

        v2 = v[i] + dt / 2 * a1
        total_force = ti.Vector(gravity) * particle_mass
        for j in range(n):
            if rest_length[i, j] != 0:
                x_ij = x[i] - x[j] + dt / 2 * v1
                v_ij = v[i] - v[j] + dt / 2 * a1
                total_force += -damping[None] * x_ij.normalized() * v_ij * x_ij.normalized()  # damping
                total_force += -spring_stiffness[None] * (x_ij.norm() - rest_length[i, j]) * x_ij.normalized()  # spring
        a2 = total_force / particle_mass

        v3 = v[i] + dt / 2 * a2
        total_force = ti.Vector(gravity) * particle_mass
        for j in range(n):
            if rest_length[i, j] != 0:
                x_ij = x[i] - x[j] + dt / 2 * v2
                v_ij = v[i] - v[j] + dt / 2 * a2
                total_force += -damping[None] * x_ij.normalized() * v_ij * x_ij.normalized()  # damping
                total_force += -spring_stiffness[None] * (x_ij.norm() - rest_length[i, j]) * x_ij.normalized()  # spring
        a3 = total_force / particle_mass

        v4 = v[i] + dt * a3
        total_force = ti.Vector(gravity) * particle_mass
        for j in range(n):
            if rest_length[i, j] != 0:
                x_ij = x[i] - x[j] + dt * v3
                v_ij = v[i] - v[j] + dt * a3
                total_force += -damping[None] * x_ij.normalized() * v_ij * x_ij.normalized()  # damping
                total_force += -spring_stiffness[None] * (x_ij.norm() - rest_length[i, j]) * x_ij.normalized()  # spring
        a4 = total_force / particle_mass

        # Compute new vel
        v[i] += dt / 6 * (a1 + 2 * a2 + 2 * a3 + a4)
        # Compute new position
        x[i] += dt / 6 * (v1 + 2 * v2 + 2 * v3 + v4)


init_mass_spring_system()

while True:
    process_input()

    if not paused[None]:
        for step in range(10):
            substep()
            collide_with_ground()
            # update_position() # rk4 已经计算了 new position
            compute_damp_energy()
        compute_current_energy()
    process_output()
