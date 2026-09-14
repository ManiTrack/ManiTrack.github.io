# ManiTrack Project Page

ManiTrack 的静态论文项目主页，基于 [Nerfies](https://nerfies.github.io/) 模板修改。

## 目录结构

```text
ManiTrack.github.io/
├── index.html
├── README.md
└── static/
    ├── css/       # 页面样式与 Bulma
    ├── images/    # favicon 与 pipeline 图片
    ├── js/        # 页面交互与图标脚本
    ├── results/   # 网页实际使用的 RGB、GLB 和 2D 结果
    └── videos/    # 后续添加的 MP4 视频
```

每个 `static/results/<hand>/<frame>/` 目录只保留五个网页资源：

- `rgb.png`
- `gt.glb`
- `gt-poster.png`
- `prediction.glb`
- `prediction-2d.png`

相关第三方许可文件保存在 `static/results/licenses/`。

## 本地预览

在本目录运行：

```powershell
python -m http.server 8000
```

然后打开 <http://localhost:8000>。也可以在 VS Code 中安装 Microsoft Live Preview，直接预览 `index.html`。

## 首屏背景

当前全屏首屏暂时使用 `static/images/pipeline.png` 作为背景，并叠加深色半透明遮罩。准备好专用背景图后，修改 `index.html` 中 `hero-background-media` 图片的 `src` 即可。首屏使用短距离 sticky 过渡；向下滚动时，标题区会明显淡出、缩小并向上收起，效果代码位于 `static/js/index.js`。

## 修改内容

1. 在 `index.html` 中替换文字、链接与媒体路径。
2. 将普通图片放入 `static/images/`，视频放入 `static/videos/`。
3. 将结果资源放入 `static/results/<hand>/<frame>/`。
4. 在 `static/css/index.css` 中调整颜色、间距和尺寸。

## LaTeX 公式

页面已经集成 MathJax 4，可直接在 `index.html` 的普通文字区域使用 LaTeX。

行内公式：

```html
The hand state is $\mathbf{q}_t \in \mathbb{R}^{D}$.
```

独立公式：

```html
\[
  \hat{\mathbf{q}}_t = \arg\min_{\mathbf{q}} \mathcal{L}(\mathbf{q})
\]
```

也支持 `\(...\)` 和 `$$...$$`。如果正文需要显示普通美元符号，请写成 `\$`。`pre` 和 `code` 中的 BibTeX/代码不会被 MathJax 处理。

## 前端依赖

- Bulma：响应式页面布局。
- Font Awesome / Academicons：论文、代码和视频按钮图标。
- `static/css/index.css`：当前项目的可编辑样式。

## License and attribution

This template is adapted from the [Nerfies website](https://nerfies.github.io/), which is shared under the [Creative Commons Attribution-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-sa/4.0/).
