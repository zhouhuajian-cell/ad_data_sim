# 仿真数据交付格式规范（锁定版）

> **本文件为交付标准的唯一依据。后续所有仿真产物必须严格按此格式输出。**
> 来源：真实已标注数据（`\\192.168.20.206\done_data_new`），逐字段核对确认。

最后验证时间：2026-11-23
验证方式：用真实 XML 的内参+3D 框投影回原图，框与真实车辆像素级贴合。

---

## 一、OBJ（3D 目标）

### 真实数据路径参考
```
\\192.168.20.206\done_data_new\18_bev-obj\02_overseas\CZ\04_songbiao\已送\
  └── 20261021_LYNK01_CZ_guizhou_1v_1_tuoche/     ← clip 名
      └── 134166-1785156623.099/                   ← <clip_id>-<timestamp>
          ├── cam_st_x2f_front/                    ← 前视相机
          │   ├── 1785156623.099.jpg
          │   └── 1785156623.099.jpg.et            ← 标记文件
          ├── lidar_hs_pandar128_topmiddle/        ← LiDAR 点云
          │   └── 1785156623.099.bin
          └── label_lidar_hs_pandar128_topmiddle/  ← 标注
              ├── 1785156623.099.xml
              └── 1785156623.099_ctrl.txt
```

### 目录规范（仿真产物）
```
<out_root>/
└── <clip_name>/
    └── <clip_id>-<timestamp>/
        ├── cam_st_x2f_front/<timestamp>.jpg
        ├── lidar_hs_pandar128_topmiddle/<timestamp>.bin
        └── label_lidar_hs_pandar128_topmiddle/
            ├── <timestamp>.xml
            └── <timestamp>_ctrl.txt        （空文件）
```

### XML Schema（61 字段，已验证 100% 一致）

```xml
<?xml version="1.0" ?>
<Lidar>
  <CaptureInfo>
    <EgoPoseParam validFlag="1">
      <wx>326832.8482847852</wx>      <!-- UTM 东向 -->
      <wy>5505369.013629333</wy>      <!-- UTM 北向 -->
      <wz>602.7794533127835</wz>      <!-- 海拔 -->
      <roll>-0.016330798057431255</roll>   <!-- 弧度 -->
      <pitch>-0.0727426608643137</pitch>
      <yaw>0.8867972571842079</yaw>
      <qx>0.008228939241525159</qx>
      <qy>-0.03634653595918279</qy>
      <qz>0.4284456630443645</qz>
      <qw>0.902798802886241</qw>
    </EgoPoseParam>
    <Lidar2CamParam>
      <cam_st_x2f_front>
        <intrinsic_matrix0>1317.960241</intrinsic_matrix0>   <!-- fx -->
        <intrinsic_matrix1>0.000000</intrinsic_matrix1>
        <intrinsic_matrix2>964.626852</intrinsic_matrix2>    <!-- cx -->
        <intrinsic_matrix3>0.000000</intrinsic_matrix3>
        <intrinsic_matrix4>1317.531714</intrinsic_matrix4>   <!-- fy -->
        <intrinsic_matrix5>540.970617</intrinsic_matrix5>    <!-- cy -->
        <intrinsic_matrix6>0.000000</intrinsic_matrix6>
        <intrinsic_matrix7>0.000000</intrinsic_matrix7>
        <intrinsic_matrix8>1.000000</intrinsic_matrix8>
        <distortion_coeffs0>-0.199256</distortion_coeffs0>   <!-- k1 -->
        <distortion_coeffs1>0.173730</distortion_coeffs1>    <!-- k2 -->
        <distortion_coeffs2>0.000053</distortion_coeffs2>    <!-- p1 -->
        <distortion_coeffs3>-0.000144</distortion_coeffs3>   <!-- p2 -->
        <distortion_coeffs4>0.049205</distortion_coeffs4>    <!-- k3 -->
        <distortion_coeffs5>0.350606</distortion_coeffs5>
        <distortion_coeffs6>-0.039546</distortion_coeffs6>
        <distortion_coeffs7>0.192720</distortion_coeffs7>
        <rotation_vector0>1.212570</rotation_vector0>        <!-- 罗德里格斯 -->
        <rotation_vector1>-1.206444</rotation_vector1>
        <rotation_vector2>1.186786</rotation_vector2>
        <translation_vector0>0.038408</translation_vector0>  <!-- 米 -->
        <translation_vector1>1.521471</translation_vector1>
        <translation_vector2>-2.083133</translation_vector2>
      </cam_st_x2f_front>
    </Lidar2CamParam>
    <VehicleInfo validFlag="0">
      <Timestamp>-999.000</Timestamp>
      <Yawrate>-999.000000</Yawrate>
      <Vehiclespeed>-999.000000</Vehiclespeed>
      <BrkPdlPos>-999.000000</BrkPdlPos>
      <WhlAng>-999.000000</WhlAng>
      <WhlAngGrd>-999.000000</WhlAngGrd>
      <AccelActuPos>-999.000000</AccelActuPos>
    </VehicleInfo>
    <MultiFrameInfo validFlag="1"/>
  </CaptureInfo>

  <LabelInfo>
    <!-- 8 个组，无对象时自闭合 Amount="0" -->
    <Three Amount="0"/>
    <Cone Amount="0"/>
    <Unknown Amount="0"/>
    <Unsure Amount="0"/>
    <Special Amount="0"/>
    <Vehicle Amount="5">
      <Object ID="0" tracking_id="0" kind="bus">
        <uuid>0b4ed288cc8611f1ad40cded696d1cb5</uuid>
        <objWx>16.573366</objWx>          <!-- 自车坐标，纵向（车头为正） -->
        <objWy>5.989583</objWy>           <!-- 自车坐标，横向（左正右负） -->
        <objWz>1.37232</objWz>            <!-- ★ 车辆中心离地高度 -->
        <objLength>6.248611</objLength>   <!-- 米 -->
        <objWidth>2.240107</objWidth>
        <objHeight>2.639517</objHeight>
        <objYaw>-2.888793</objYaw>        <!-- 弧度 -->
        <objVelocityX>-9.4316</objVelocityX>
        <objVelocityY>-1.8066</objVelocityY>
        <objVelocityZ>-0.2242</objVelocityZ>
        <objRoll>0.0</objRoll>
        <objPitch>0.0</objPitch>
        <objCertainty>0.981709</objCertainty>
        <motion>1</motion>
        <lidarPtsCnt>4047</lidarPtsCnt>
        <inVisible>1</inVisible>
      </Object>
    </Vehicle>
    <Rider Amount="0"/>
    <Pedestrian Amount="1">
      <Object ID="0" tracking_id="3">
        ... 同上字段 ...
        <crowd>0</crowd>                  <!-- ★ Pedestrian 专有 -->
      </Object>
    </Pedestrian>
  </LabelInfo>
</Lidar>
```

### 关键约定（已用真实数据验证）

| 项 | 结论 | 验证方式 |
|---|---|---|
| `objWx` | 自车坐标系**纵向**距离（米） | R 矩阵第3行 `[0.9999, 0.0161, -0.0052]` |
| `objWy` | 自车坐标系**横向**位置（左正） | 投影后左右位置正确 |
| `objWz` | **车辆中心离地高度**（= H/2） | 框底边精确贴地 |
| `objYaw` | 弧度 | 框朝向正确 |
| 变换 | `p_cam = R @ p_body + T`，R 由 rotation_vector 罗德里格斯展开 | 投影像素级贴合 |
| K | 直接使用 `intrinsic_matrix0..8` | — |

### 组与类别映射

```
Three       ← tricycle / three
Cone        ← cone / traffic_cone
Vehicle     ← car/sedan/suv/taxi/police/van/truck/bus（kind 属性区分）
Rider       ← bicycle / motorcycle / scooter / cyclist / rider
Pedestrian  ← pedestrian / person（额外 crowd 字段）
Special     ← 其他
Unknown / Unsure  ← 不确定
```

### 导出脚本
`tools/export_to_label.py`

---

## 二、TSR（交通标志）

### 真实数据路径参考
```
\\192.168.20.206\done_data_new\05_tlts\TLTS_overseas\auto_train_done\
  20261111_overseas_tsr_DE_simulation_148\DE\
  └── 20260809_LYNK01_DE_2M_night_TLTS_release_slow_down_rider/
      ├── 1786220954.649.jpg
      ├── 1786220954.649.xml            ← 同名配对
      ├── 1786220955.849.jpg
      ├── 1786220955.849.xml
      ...（37 组）
      └── *.db
```

### 目录规范
```
<out_root>/
└── <clip_name>/
    ├── <timestamp>.jpg
    └── <timestamp>.xml
```
**注意：TSR 无子目录，图片与 XML 平铺同名配对。**

### XML Schema（极简但语义密集）

```xml
<?xml version='1.0' encoding='utf-8'?>
<Image>
  <LabelInfo>
    <Vehicle Amount="0" />
    <Pedestrian Amount="0" />
    <Rider Amount="0" />
    <TrafficSign Amount="1">
      <Object shape="CIRCLE"
              firstAttribute="LIMIT"
              color="UNKNOWN"
              isCovered="Level0"
              isVirtualScene="false"        <!-- ★ 仿真应设 true -->
              ID="1"
              deep="0"
              definition="obvious"
              precision="float"
              labelType="POS"
              fValue="0.00"
              LimitType="SPEED"
              type="RELEASE"
              LimitTypeOverseas="TrafficReduce"
              trafficReduceSubType="NonMotorizedArea"
              edgePts00X="514.33" edgePts00Y="377.7"
              edgePts01X="582.83" edgePts01Y="445.67" />
    </TrafficSign>
  </LabelInfo>
</Image>
```

### 字段含义

| 字段 | 示例 | 含义 |
|---|---|---|
| `shape` | `CIRCLE` / `TRIANGLE` / `RECTANGLE` | 标志形状 |
| `firstAttribute` | `LIMIT` / `WARNING` / `PROHIBIT` | 一级属性 |
| `color` | `UNKNOWN` / `BLUE` / `RED` / `YELLOW` | 底色 |
| `isCovered` | `Level0` / `Level1`... | 遮挡等级 |
| `isVirtualScene` | `false` / `true` | **仿真数据设 true** |
| `ID` | `1` | 实例 ID |
| `deep` | `0` | 远近 |
| `definition` | `obvious` | 清晰度 |
| `precision` | `float` | 精度 |
| `labelType` | `POS` / `NEG` | 正/负样本 |
| `fValue` | `0.00` | 数值（限速值等） |
| `LimitType` | `SPEED` | 限速类型 |
| `type` | `RELEASE` | 类型 |
| `LimitTypeOverseas` | `TrafficReduce` | 海外子类 |
| `trafficReduceSubType` | `NonMotorizedArea` | 更细子类 |
| `edgePts00X/Y` | 514.33, 377.7 | **框角点 0（左上）** |
| `edgePts01X/Y` | 582.83, 445.67 | **框角点 1（右下）** |

### 关键约定（待补充验证）
- **bbox 只有 2 个对角点**（edgePts00 = 左上，edgePts01 = 右下），不是 4 点
- 需扫描全部 37 个 XML 确认：负样本格式、三角/矩形标志、是否需要 4 点

---

## 三、GND（地面标线）

### 真实数据路径参考
```
\\192.168.20.206\done_data_new\31_ex_GND\05_maxieye\05_MG6_002\20260928\
  └── 1790569917.088/                       ← 直接是 timestamp 目录
      ├── img_ori/                          ← 原始图像 jpg
      ├── img_rec/                          ← ★ 重建/仿真图 jpg
      ├── seg_lane/                         ← ★ 车道线标注 png（调色板）
      ├── freespace_mask_ori_3.0.0/         ← 可行驶区域 png
      ├── obj_mask_ori_3.1.1/               ← 车辆 mask png
      ├── sky_mask_ori_3.0.0/               ← 天空 mask png
      ├── top/                              ← 俯视点云 .las
      ├── pose_imu/                         ← 位姿 .las
      ├── IMU/
      └── parameter_yaml/<date>/
          ├── intrinsic/camera-1.yaml
          └── extrinsic/{camera-1_to_body-2,lidar-1_to_body-2}.yaml
```

### 目录规范（仿真产物）
```
<out_root>/
└── <timestamp>/
    ├── img_ori/<timestamp>.jpg            （原图，直接复制）
    ├── img_rec/<timestamp>.jpg            （★ 仿真图）
    ├── seg_lane/<timestamp>.png           （★ 标注）
    └── parameter_yaml/<date>/...
```

### seg_lane 调色板（RGB，1920×1080）

```
(0, 0, 0)         背景
(0, 255, 0)       车道边线 edge
(232, 35, 244)    虚线/标线 dash
(153, 153, 190)   远景 far
```

### 关键约定
- **标注是调色板 PNG mask，不是 JSON/XML**
- `img_rec/` 放仿真图（算法重建结果）
- `parameter_yaml/` 提供相机内外参（独立 yaml，非内嵌）

---

## 四、脚本清单

| 模块 | 导出脚本 | 状态 |
|---|---|---|
| OBJ | `tools/export_to_label.py` | ✅ 已验证 61/61 字段 |
| TSR | 待写 | ⏳ 需扫描全部样本确认边缘情况 |
| GND | 待写 | ⏳ 需确认 parameter_yaml 格式 |

---

## 五、验证记录

### OBJ objWz 语义验证（2026-11-23）

用真实 XML 的 `intrinsic_matrix` + `rotation_vector`/`translation_vector`，
把 `Object` 的 8 个 3D 角点投影回原图：

```
输入: 20261021_LYNK01_CZ_guizhou_1v_1_tuoche/134166-1785156623.099
     label_lidar_hs_pandar128_topmiddle/1785156623.099.xml
     + cam_st_x2f_front/1785156623.099.jpg

结果:
  Vehicle bus  x=16.6 y=6.0 z=1.4  L=6.2 W=2.2 H=2.6  uv0=(530,407)  ✅ 贴合
  Vehicle bus  x=41.6 y=7.7 z=1.9  L=20.2 W=3.2 H=4.1  uv0=(567,470)  ✅ 贴合

结论: objWx=纵向 objWy=横向 objWz=中心离地高度，全部正确。
      p_cam = R @ p_body + T （R 由旋转向量罗德里格斯展开）
```

验证脚本：`tools/verify_obj_boxes.py`
