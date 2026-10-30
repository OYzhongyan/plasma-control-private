# 托卡马克等离子体形状参数

![tokamak_shape_params.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/Yvenve5B8YKAMloy/img/99fa5e4e-9724-4fb7-8a73-37bbb6adf247.png)

在托卡马克装置中，等离子体的极向截面（$R$\-$Z$ 平面）近似为拉长的 D 形。上图中蓝色曲线为**最外闭合磁面（LCFS）**，其形状由以下参数描述。

本文主要对应 `docs/reference.md` 中的 `Rmax`、`Rmin`、`aminor`、`kappa`、`deltal`、`deltau`、`rc`、`zc` 等 observation 字段，便于把几何图示和环境字段一一对应起来。

## 辅助几何量

从 LCFS 边界可直接读取以下几何量（见图中标注点）：

| 符号 | 含义 |
| --- | --- |
| $R\_{\max}$ | LCFS 在中平面（$Z=0$<br>）上的**最大**大半径 |
| $R\_{\min}$ | LCFS 在中平面上的**最小**大半径 |
| $b$ | LCFS 的半高度，即最高点的 <br>坐标 |
| $R\_{\mathrm{top}}$ | LCFS 最高点对应的 <br>坐标 |
| $R\_{\mathrm{bot}}$ | LCFS 最低点对应的 <br>坐标 |

说明：

- 这里的 `R` 是托卡马克柱坐标中的大半径方向，`Z` 是垂直方向
- `Rmax` / `Rmin` 决定了截面的左右边界范围
- `Rtop` / `Rbot` 则决定了上、下极值点相对几何中心的横向偏移

## 形状参数定义

**几何中心大半径**与**小半径**：

$R\_{\mathrm{mid}} = \frac{R\_{\max} + R\_{\min}}{2}, \qquad a = \frac{R\_{\max} - R\_{\min}}{2}$

在环境字段中，可近似对应为：

- `aminor = a`
- `rc` 通常可理解为几何中心对应的半径量，和 $R_{\mathrm{mid}}$ 含义接近
- `zc` 为几何中心的垂直坐标；若位形上下不完全对称，`zc` 不一定为 0

**拉长比**（elongation）：

$\kappa = \frac{b}{a}$

$\kappa = 1$对应圆形截面，$\kappa > 1$ 表示截面沿 $Z$ 方向拉长。

**上三角形变**（upper triangularity）与**下三角形变**（lower triangularity）：

$\delta\_U = \frac{R\_{\mathrm{mid}} - R\_{\mathrm{top}}}{a}, \qquad \delta\_L = \frac{R\_{\mathrm{mid}} - R\_{\mathrm{bot}}}{a}$

$\delta > 0$表示极值点相对几何中心向内侧（小 $R$ 方向）偏移，截面呈 D 形。

在 observation 中通常对应：

- `deltau = \delta_U`
- `deltal = \delta_L`

可这样直观理解：

- `kappa` 越大，等离子体越“高瘦”
- `deltau` 越大，上半部越向内收
- `deltal` 越大，下半部越向内收
- 当 `deltau` 与 `deltal` 差异较大时，说明上下半部形状不对称更明显