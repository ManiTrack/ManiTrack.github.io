# Academic Project Page

这是一个基于 [Nerfies](https://nerfies.github.io/) 清理得到的空白论文项目主页模板。

## 本地预览

在本目录运行：

```powershell
python -m http.server 8000
```

然后打开 <http://localhost:8000>。也可以在 VS Code 中安装 Microsoft Live Preview，直接预览 `index.html`。

## 首屏背景

当前全屏首屏暂时使用 `static/images/pipeline.png` 作为背景，并叠加深色半透明遮罩。准备好专用背景图后，修改 `index.html` 中 `hero-background-media` 图片的 `src` 即可。首屏使用短距离 sticky 过渡；向下滚动时，标题区会明显淡出、缩小并向上收起，效果代码位于 `static/js/index.js`。

## 填写顺序

1. 在 `index.html` 中搜索 `[`，依次替换标题、作者、单位、摘要和 BibTeX。
2. 将图片放入 `static/images/`，视频放入 `static/videos/`。
3. 用 `<img>`、`<video>` 或 YouTube `<iframe>` 替换页面里的灰色媒体占位块。
4. 在 `static/css/index.css` 中调整颜色、间距和尺寸。
5. 将按钮中的 `href="#"` 替换为论文、代码、视频和数据链接。

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

## 保留的依赖

- Bulma：响应式页面布局。
- Font Awesome / Academicons：论文、代码和视频按钮图标。
- `static/css/index.css`：当前项目的可编辑样式。

原始网站的论文内容、Analytics、轮播、插值脚本和演示素材已经移除。

## License and attribution

This template is adapted from the [Nerfies website](https://nerfies.github.io/), which is shared under the [Creative Commons Attribution-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-sa/4.0/).
