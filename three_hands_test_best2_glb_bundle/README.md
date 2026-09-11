# Three-hand test best-two GLB bundle

这是 Inspire、XHand、LEAP Hand 各两个优质预测样本的完整自包含包。

## 选择原则

- 每种手型使用同一组 0704 final Stage-3 完整测试报告（各 1000 帧）。
- 按 `mean_gt_3d_error_m` 从小到大排序，直接选择各手型前两名。
- 六个入选样本都已目视确认手部清晰；报告保存的 GT 与本包当前 test GT 对齐误差小于 0.0001 mm。
- XHand test 的 link/joint 是 `right_hand_*`，因此使用右手 URDF。

| Hand | Rank | Frame | Mean GT 3D error | Reprojection | Depth anchors | Objects |
|---|---:|---:|---:|---:|---:|---:|
| inspire | 1 | 105 | 2.5664 mm | 3.843 px | 8 | 0 |
| inspire | 2 | 327 | 2.7620 mm | 5.404 px | 13 | 1 |
| xhand | 1 | 502 | 3.6877 mm | 5.466 px | 10 | 2 |
| xhand | 2 | 89 | 3.7087 mm | 6.015 px | 9 | 0 |
| leap | 1 | 966 | 3.1339 mm | 8.093 px | 14 | 1 |
| leap | 2 | 335 | 3.7029 mm | 6.438 px | 14 | 2 |

## 文件布局

- `{hand}/frame_XXXXXX/source/`：RGB、Depth、当前 GT 场景数据、精简后的模型预测数据。
- `{hand}/frame_XXXXXX/models/gt.glb`：GT 手姿 + 该帧记录物体。
- `{hand}/frame_XXXXXX/models/prediction.glb`：预测手姿 + 同一记录物体。
- `{hand}/frame_XXXXXX/previews/`：透明 GT/预测渲染、预测 2D 关键点骨架、四列总览。
- `assets/`：仅保留运行所需的三套视觉 URDF 和原始视觉 mesh/material。
- `scripts/`：从任意工作目录重生成和验证的脚本。

两个 GLB 都不包含背景；mesh、材质和几何数据已写入 GLB，不含外部 URI。若该帧有记录物体，GT 与预测 GLB 使用完全相同的物体尺寸、位姿、颜色、粗糙度和金属度。

## 重生成与验证

从任意目录执行（把 `/path/to/...` 替换为复制后的包路径）：

```bash
python /path/to/three_hands_test_best2_glb_bundle/scripts/regenerate_all.py
python /path/to/three_hands_test_best2_glb_bundle/scripts/verify_bundle.py
```

重生成只访问本包内文件，不读取 DexTrack 仓库、原 test 数据或原 checkpoint。`validation_report.json` 为结构、数值、GLB 内嵌资源与清单校验结果；`bundle_manifest.json` 记录文件 SHA256。
