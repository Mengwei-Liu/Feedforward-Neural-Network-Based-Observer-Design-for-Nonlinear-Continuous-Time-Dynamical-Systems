import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import os

#---------------------------------------
# 参数
#---------------------------------------
L_ti = 1
C_i = 0.05
R_i = 96
P_i = 0.1
EPS = 0.1   # 防止除零导致刚性爆炸

def u(t):
    return np.sin(t)

def system(t, z):
    I_ti, V_i = z
    dI_ti = (-V_i + u(t)) / L_ti
    dV_i = (I_ti - 0.1 - (1/R_i)*V_i )/ C_i
    return [dI_ti, dV_i]

#---------------------------------------
# 仿真配置
#---------------------------------------
t_total = 20           # 总时间 20 s
dt_segment = 2         # 每段 2 s
num_segments = int(t_total / dt_segment)
t_points_per_segment = 20000   # 每段 20000 点（原来2000）
sample_step = 2        # 每隔 2 点采样（原来10）

initial_conditions = [
    [0, 100],
    [-1, 150],
    [3, 80],
    [3, 150],
    [-1, 100],
    [2, 80],
    [0, 150],
    [0, 80],

]

output_file = "system_samples.csv"
if os.path.exists(output_file):
    os.remove(output_file)

#---------------------------------------
# 分段积分 + 实时保存
#---------------------------------------
all_samples = []

for idx, z0 in enumerate(initial_conditions):
    print(f"\n🔹 开始积分第 {idx+1} 组初值: {z0}")
    current_z = np.array(z0, dtype=float)

    for seg in range(num_segments):
        t_start = seg * dt_segment
        t_end = (seg + 1) * dt_segment
        t_eval = np.linspace(t_start, t_end, t_points_per_segment)

        sol = solve_ivp(system, (t_start, t_end), current_z,
                        method="BDF", t_eval=t_eval,
                        rtol=1e-4, atol=1e-6)

        if not sol.success:
            print(f"⚠️ 积分失败：段 {seg+1}, 初值 {z0}")
            break

        current_z = sol.y[:, -1]
        for i in range(0, len(sol.t), sample_step):
            I_val = sol.y[0][i]
            V_val = sol.y[1][i]

            if (-1 < I_val < 3) and (80 < V_val < 150):
                all_samples.append([I_val, V_val])

        # 实时保存
        if len(all_samples) >= 5000:
            df = pd.DataFrame(all_samples, columns=["I_ti", "V_i"])
            df.to_csv(output_file, mode="a", index=False, header=not os.path.exists(output_file))
            all_samples = []
            print(f"  ✅ 已保存到文件：t = {t_end:.1f} 秒")

    print(f"  ✅ 完成初值 {z0} 全程积分")

# 剩余数据保存
if len(all_samples) > 0:
    df = pd.DataFrame(all_samples, columns=["I_ti", "V_i"])
    df.to_csv(output_file, mode="a", index=False, header=not os.path.exists(output_file))
    print("  ✅ 剩余数据已保存")

print("\n🎉 全部积分完成，文件已保存为：", os.path.abspath(output_file))



