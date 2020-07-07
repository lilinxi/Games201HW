import taichi as ti
import numpy as np

ti.init(arch=ti.cpu)

resolution = 512
gui = ti.GUI('Mass-Spring System', res=(resolution, resolution), background_color=0xdddddd)

max_num_particles = 500  # 最大粒子数目

dt = 1e-3  # 时间步长

num_particles = ti.var(ti.i32, shape=())  # 质点数目
spring_stiffness = ti.var(ti.f32, shape=())  # 弹簧硬度
paused = ti.var(ti.i32, shape=())  # 是否暂停
damping = ti.var(ti.f32, shape=())  # 阻尼

particle_mass = 1  # 质点质量
bottom_y = 0.05  # 最底部y

x = ti.Vector(2, dt=ti.f32, shape=max_num_particles)  # 质点位置
v = ti.Vector(2, dt=ti.f32, shape=max_num_particles)  # 质点速度

# rest_length[i, j] = 0 means i and j are not connected
rest_length = ti.var(ti.f32, shape=(max_num_particles, max_num_particles))  # 两个质点之间的弹簧原长

connection_radius = 0.15  # 两个质点连接的最大距离

gravity = [0, -9.8]  # 重力

# 计算能量
origin_energy = ti.var(ti.f32, shape=())  # 原始的能量，只有重力势能
current_energy = ti.var(ti.f32, shape=())  # 现在的能量，重力势能+弹性势能+动能
lost_energy = ti.var(ti.f32, shape=())  # 损失的能量，撞击地面损失的动能，转化为了热能
damp_energy = ti.var(ti.f32, shape=())  # 损失的能量，阻尼损失的动能，转化为了热能


@ti.kernel
def new_particle(pos_x: ti.f32, pos_y: ti.f32):
    new_particle_id = num_particles[None]  # 必须用None访问标量
    x[new_particle_id] = [pos_x, pos_y]
    v[new_particle_id] = [0, 0]
    num_particles[None] += 1
    origin_energy[None] += -particle_mass * gravity[1] * pos_y  # mgh

    # Connect with existing particles
    for i in range(new_particle_id):
        dist = (x[new_particle_id] - x[i]).norm()
        if dist < connection_radius:
            rest_length[i, new_particle_id] = dist
            rest_length[new_particle_id, i] = dist


# collide with ground
@ti.kernel
def collide_with_ground():
    for i in range(num_particles[None]):
        if x[i].y < bottom_y:
            x[i].y = bottom_y  # 修正位置会引起损失能量的计算误差变大
            if v[i].y < 0:  # 只计算和置零向下的速度
                # print(v[i].y, i)
                lost_energy[None] += 0.5 * particle_mass * v[i].y * v[i].y
                v[i].y = 0


# compute new position
@ti.kernel
def update_position():
    for i in range(num_particles[None]):
        # print(1,x[i], v[i])
        if x[i].y <= bottom_y:
            print(x[i]) # 这个print去掉在我的电脑上隐式方法的就跑不了，未解之谜
            v[i].y = 0
        # print(i, v[i])
        # print("b:",x[i])
        # print(2,x[i], v[i])
        x[i] += v[i] * dt
        # print(3,x[i], v[i])
        # print("f:",x[i])


# (green <-- black --> red)
# red means the spring is elongating
# green means the spring is compressing
@ti.kernel
def calculate_color(delta: ti.f32) -> ti.i32:
    eps = 0.00001
    color = 0x445566
    if delta > eps:
        color = 0xFF0000
    elif delta < -eps:
        color = 0x00FF00
    return color
    # sigmoid = 2 / (1 + ti.exp(-delta * spring_stiffness[None] * 0.1)) - 1
    # return int(max(sigmoid, 0) * 0xff) * 0x10000 - int(min(sigmoid, 0) * 0xff) * 0x100


@ti.kernel
def compute_current_energy():  # Compute current energy
    current_energy[None] = 0
    n = num_particles[None]
    for i in range(n):
        # 重力势能
        current_energy[None] += -particle_mass * gravity[1] * x[i][1]
        # 动能
        current_energy[None] += 0.5 * particle_mass * v[i].norm() * v[i].norm()
        # 弹性势能
        for j in range(i, n):  # 防止计算两次
            if rest_length[i, j] != 0:
                x_ij = x[i] - x[j]
                # 1/2 k x^2
                current_energy[None] += 0.5 * spring_stiffness[None] * (x_ij.norm() - rest_length[i, j]) ** 2


@ti.kernel
def compute_damp_energy():
    n = num_particles[None]
    for i in range(n):
        total_force = ti.Vector([0, 0])
        for j in range(n):
            if rest_length[i, j] != 0:
                x_ij = x[i] - x[j]
                v_ij = v[i] - v[j]
                total_force += -damping[None] * x_ij.normalized() * v_ij * x_ij.normalized()  # damping
        damp_energy[None] += abs(total_force.dot(v[i]) * dt)


def init_mass_spring_system():
    # initial parameter status
    spring_stiffness[None] = 10000
    # spring_stiffness[None] = 1000000
    damping[None] = 20

    new_particle(0.3, 0.3)
    new_particle(0.3, 0.4)
    new_particle(0.4, 0.4)


def process_input():
    for e in gui.get_events(ti.GUI.PRESS):
        if e.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]:
            exit()
        elif e.key == gui.SPACE:
            paused[None] = not paused[None]
        elif e.key == ti.GUI.LMB:
            new_particle(e.pos[0], e.pos[1])
        elif e.key == 'c':
            num_particles[None] = 0
            rest_length.fill(0)
        elif e.key == 's':
            if gui.is_pressed('Shift'):
                spring_stiffness[None] /= 1.1
            else:
                spring_stiffness[None] *= 1.1
        elif e.key == 'd':
            if gui.is_pressed('Shift'):
                damping[None] /= 1.1
            else:
                damping[None] *= 1.1


def process_output():
    X = x.to_numpy()
    gui.circles(X[:num_particles[None]], color=0xffaa77, radius=5)

    gui.line(begin=(0.0, bottom_y), end=(1.0, bottom_y), color=0x0, radius=1)

    for i in range(num_particles[None]):
        for j in range(i + 1, num_particles[None]):
            if rest_length[i, j] != 0:
                norm = np.linalg.norm(X[i] - X[j])
                gui.line(begin=X[i], end=X[j], radius=2, color=calculate_color(norm - rest_length[i, j]))
    gui.text(content=f'C: clear all; Space: pause', pos=(0, 0.95), color=0x0)
    gui.text(content=f'S: Spring stiffness {spring_stiffness[None]:.1f}', pos=(0, 0.9), color=0x0)
    gui.text(content=f'D: damping {damping[None]:.2f}', pos=(0, 0.85), color=0x0)
    gui.text(content=f'Number of particles {num_particles[None]:.0f}', pos=(0, 0.80), color=0x0)
    gui.text(content=f'Origin energy {origin_energy[None]:.0f}', pos=(0, 0.75), color=0x0)
    gui.text(content=f'Current energy {current_energy[None]:.0f}', pos=(0, 0.70), color=0x0)
    gui.text(content=f'Lost energy {lost_energy[None]:.0f}', pos=(0, 0.65), color=0x0)
    gui.text(content=f'Damp energy {damp_energy[None]:.0f}', pos=(0, 0.60), color=0x0)
    gui.text(content=f'Total energy {current_energy[None] + lost_energy[None] + damp_energy[None]:.0f}',
             pos=(0, 0.55), color=0x0)
    gui.text(
        content=f'Error energy {origin_energy[None] - current_energy[None] - lost_energy[None] - damp_energy[None]:.0f}',
        pos=(0, 0.50), color=0x0)
    gui.show()

# @ti.kernel
# def paint():
#     for i, j, k in pixels:
#         pixels[i, j, k] = ti.random() * 255
#
#
# def process_gif():
#     result_dir = "./results"
#     video_manger = ti.VideoManager(output_dir=result_dir, framerate=24, automatic_build=False)
#
#     for i in range(50):
#         paint()
#
#         pixels_img = pixels.to_numpy()
#         video_manger.write_frame(pixels_img)
#         print(f'\rFrame {i + 1}/50 is recorded', end='')
#
#     print()
#     print('Exporting .mp4 and .gif videos...')
#     video_manger.make_video(gif=True, mp4=True)
#     print(f'MP4 video is saved to {video_manger.get_output_filename(".mp4")}')
#     print(f'GIF video is saved to {video_manger.get_output_filename(".gif")}')
