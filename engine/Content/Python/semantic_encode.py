# semantic_encode.py — 语义通道的「编码表」与生成/解码工具
#
# 为什么需要编码：
#   实测把 Emissive 设为 class_id/255（如 1/255=0.0039）时，输出会被抬到 13~50 区间
#   （gamma + 曝光），低值区完全不可用，无法还原出 1~7 这样的小整数。
#   所以改成「把类别映射到分离度足够的亮度档位」，再用阈值解码。
#
# 标定数据（road 材质常量 → 捕获输出，AEM_MANUAL 固定曝光）：
#   输入 1→13, 2→22, 4→34, 8→50, 16→71, 32→99, 64→137, 128→188, 200→229, 255→255
#   据此反解出每个类别应写入的材质常量值（见 ENCODE_INPUT）。
#
# 输出值 → 类别 的判定用「最近档位」而不是「除以 255」，对 gamma 与抗锯齿都更稳。
SEMANTIC_CLASSES = {
    1: "road",
    2: "vehicle",
    3: "pedestrian",
    4: "traffic_sign",
    5: "cone",
    6: "vegetation",
    7: "landline",
}

# 目标输出档位（0~255）：均匀铺开，避开最低的不可用区间
TARGET_OUTPUT = {1: 32, 2: 64, 3: 96, 4: 128, 5: 160, 6: 192, 7: 224}

# 反解出的材质常量（0~1，即 in/255），由标定曲线查得
ENCODE_INPUT = {1: 4, 2: 13, 3: 30, 4: 56, 5: 93, 6: 135, 7: 191}

# 解码用的参考点（实测输出，0~255）
CALIBRATION = [(1, 13), (2, 22), (4, 34), (8, 50), (16, 71), (32, 99),
               (64, 137), (128, 188), (200, 229), (255, 255)]


def material_gray(class_id):
    """该类别的材质常量值（0~1）。"""
    return ENCODE_INPUT[class_id] / 255.0


def decode_value(v255):
    """把单个像素值（0~255）解码成类别 id。

    用「与各目标档位最近」判定；比除以 255 更耐受 gamma 偏差与边缘抗锯齿。
    背景（天空）输出接近 0，直接判 0。
    """
    if v255 < 20:
        return 0
    best_cls, best_err = 0, 1e9
    for cls, tgt in TARGET_OUTPUT.items():
        err = abs(v255 - tgt)
        if err < best_err:
            best_err, err_cls = err, cls
            best_cls = err_cls
    return best_cls


def decode_image(arr255):
    """整帧解码：arr255 是 HxW 的 uint 数组（0~255），返回 HxW 的类别 id 数组。

    注意：为保持依赖最小，这里用纯 Python + 常量表；调用方一般用 numpy 向量化版本
    （见 docs 里的 decode_numpy 示例）。本函数用于小图或单元测试。
    """
    h = len(arr255)
    w = len(arr255[0]) if h else 0
    out = [[0] * w for _ in range(h)]
    for y in range(h):
        row = arr255[y]
        orow = out[y]
        for x in range(w):
            orow[x] = decode_value(row[x])
    return out


if __name__ == "__main__":
    # 自检：编码表是否单调、档位间隔是否够大
    outs = [TARGET_OUTPUT[c] for c in sorted(TARGET_OUTPUT)]
    gaps = [outs[i + 1] - outs[i] for i in range(len(outs) - 1)]
    print("类别编码表：")
    for c in sorted(SEMANTIC_CLASSES):
        print("  class %d %-14s 材质常量 %.4f 目标输出 %3d"
              % (c, SEMANTIC_CLASSES[c], material_gray(c), TARGET_OUTPUT[c]))
    print("相邻档位间隔：", gaps, "最小", min(gaps))
    assert min(gaps) > 20, "档位间隔太小，解码容易出错"
    # 解码自检：每个目标值应解回自己
    for c, tgt in TARGET_OUTPUT.items():
        assert decode_value(tgt) == c, "解码失败 class %d" % c
    assert decode_value(0) == 0
    print("解码自检通过")
