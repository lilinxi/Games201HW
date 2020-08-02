import taichi as ti

ti.init(arch=ti.cpu)

dim = 2
n_particles = 8192
n_grid = 32
dx = 1 / n_grid
inv_dx = 1 / dx
dt = 2.0e-3
use_apic = False

x = ti.Vector(dim, dt=ti.f32, shape=n_particles)  # 粒子的位置
v = ti.Vector(dim, dt=ti.f32, shape=n_particles)  # 粒子的速度
C = ti.Matrix(dim, dim, dt=ti.f32, shape=n_particles)  # 粒子的 C 矩阵（APIC 计算）
grid_v = ti.Vector(dim, dt=ti.f32, shape=(n_grid, n_grid))  # 速度场
grid_m = ti.var(dt=ti.f32, shape=(n_grid, n_grid))  # 质量场


@ti.func
def clamp_pos(pos):
    return ti.Vector([max(min(0.95, pos[0]), 0.05), max(min(0.95, pos[1]), 0.05)])


@ti.kernel
def substep_PIC():
    # TODO step1. p2g
    for p in x:  # random x: [0.2,0.8]
        # TODO step1.1. 确定所在的左下角 grid
        base = (x[p] * inv_dx - 0.5).cast(int)
        # [0.5/32=.015625, 1.5/32=.046875) -> 0; [31.5/32, 32.5/32) -> 31
        # print(x[p], "->", x[p] * inv_dx, "->", base)
        # TODO step1.2. 计算左下角 grid 取整时的偏移 -> [0.5, 1.5)
        fx = x[p] * inv_dx - base.cast(float)
        # print(x[p] * inv_dx, "->", base, "->", fx)
        # TODO step1.3. 使用二次核函数根据 grid 内偏移计算 weight 权重，分别为远，近，远的三个权重
        # Quadratic B-spline
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
        # TODO step1.4. 对周围的 9 个 grid 有影响
        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                # TODO step1.4.1. 获取影响的 grid 偏移
                offset = ti.Vector([i, j])
                # TODO step1.4.2. 计算影响权重
                weight = w[i][0] * w[j][1]
                # TODO step1.4.3. 计算影响速度
                grid_v[base + offset] += weight * v[p]
                # TODO step1.4.4. 计算影响质量
                grid_m[base + offset] += weight
    # TODO step2. g op
    for i, j in grid_m:
        if grid_m[i, j] > 0:
            # TODO step2.1. grid_V = grid_V / grid_M，对每个 grid 中的速度取平均，这个平均导致 PIC 损失了大量的能量
            inv_m = 1 / grid_m[i, j]
            grid_v[i, j] = inv_m * grid_v[i, j]

    # TODO step3. g2p & p op
    for p in x:
        # TODO step3.1. 计算 grid 索引，三个 weight 权重
        base = (x[p] * inv_dx - 0.5).cast(int)
        fx = x[p] * inv_dx - base.cast(float)
        # Quadratic B-spline
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
        # TODO step3.2. 由 9 个 grid 确定新速度
        new_v = ti.Vector.zero(ti.f32, 2)
        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                # TODO step3.2.1. 叠加每个 grid 的速度
                offset = ti.Vector([i, j])
                weight = w[i][0] * w[j][1]
                new_v += weight * grid_v[base + offset]

        # TODO step3.3. 计算位置，并对位置进行 clamp，同时更新速度
        x[p] = clamp_pos(x[p] + v[p] * dt)
        v[p] = new_v


@ti.kernel
def substep_APIC():
    for p in x:
        base = (x[p] * inv_dx - 0.5).cast(int)
        fx = x[p] * inv_dx - base.cast(float)
        # Quadratic B-spline
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
        affine = C[p]  # TODO new in APIC
        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                offset = ti.Vector([i, j])
                weight = w[i][0] * w[j][1]
                dpos = (offset.cast(float) - fx) * dx  # TODO new in APIC -> [0,2] - [0.5, 1.5) / 32 = (-1.5,1.5] / 32
                grid_v[base + offset] += weight * (v[p] + affine @ dpos)  # TODO new in APIC 速度会根据 C 和 dpos 进行修正
                grid_m[base + offset] += weight

    for i, j in grid_m:
        if grid_m[i, j] > 0:
            inv_m = 1 / grid_m[i, j]
            grid_v[i, j] = inv_m * grid_v[i, j]

    for p in x:
        base = (x[p] * inv_dx - 0.5).cast(int)
        fx = x[p] * inv_dx - base.cast(float)
        # Quadratic B-spline
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
        new_v = ti.Vector.zero(ti.f32, 2)
        new_C = ti.Matrix.zero(ti.f32, 2, 2)  # TODO new in APIC 初始化 C
        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                offset = ti.Vector([i, j])
                g_v = grid_v[base + offset]
                weight = w[i][0] * w[j][1]
                new_v += weight * g_v
                # TODO new in APIC
                dpos = offset.cast(float) - fx  # TODO new in APIC
                new_C += 4 * weight * g_v.outer_product(dpos) * inv_dx  # TODO new in APIC 更新 C

        x[p] = clamp_pos(x[p] + new_v * dt)
        v[p] = new_v
        C[p] = new_C  # TODO new in APIC


@ti.kernel
def reset(mode: ti.i32):
    for i in range(n_particles):
        x[i] = [ti.random() * 0.6 + 0.2, ti.random() * 0.6 + 0.2]
        if mode == 0:
            v[i] = [1, 0]
        elif mode == 1:
            v[i] = [x[i][1] - 0.5, 0.5 - x[i][0]]
        elif mode == 2:
            v[i] = [0, x[i][0] - 0.5]
        else:
            v[i] = [0, x[i][1] - 0.5]


# 初始化为旋转，旋转最能体现 PIC 的能量损失
reset(1)

gui = ti.GUI("PIC v.s. APIC", (512, 512))
for frame in range(2000000):
    if gui.get_event(ti.GUI.PRESS):
        if gui.event.key == 't':
            reset(0)
        elif gui.event.key == 'r':
            reset(1)
        elif gui.event.key == 's':
            reset(2)
        elif gui.event.key == 'd':
            reset(3)
        elif gui.event.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]:
            break
        elif gui.event.key == 'a':
            use_apic = not use_apic
    for s in range(10):
        grid_v.fill([0, 0])
        grid_m.fill(0)
        if use_apic:
            substep_APIC()
        else:
            substep_PIC()
    scheme = 'APIC' if use_apic else 'PIC'
    gui.clear(0x112F41)
    gui.text('(D) Reset as dilation', pos=(0.05, 0.25))
    gui.text('(T) Reset as translation', pos=(0.05, 0.2))
    gui.text('(R) Reset as rotation', pos=(0.05, 0.15))
    gui.text('(S) Reset as shearing', pos=(0.05, 0.1))
    gui.text(f'(A) Scheme={scheme}', pos=(0.05, 0.05))
    gui.circles(x.to_numpy(), radius=3, color=0x068587)
    gui.show()
