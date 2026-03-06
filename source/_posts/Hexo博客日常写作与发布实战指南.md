---
title: Hexo博客日常写作与发布实战指南
date: 2026-03-07 12:00:00
updated: 2026-03-07 12:00:00
tags: [Hexo, 博客, GitHub Pages, 教程]
categories: [技术分享]
---

这篇文章给你一套可直接执行的 Hexo 日常写作与发布流程，目标是：

- 新文章能快速本地预览；
- 发布步骤固定，不容易漏；
- 出错时知道先查哪里。

文中的命令基于你当前项目结构（源码分支 `hexo`，部署分支 `main`，`hexo-deployer-git` 发布到 GitHub Pages）。

## 适用人群

- 已经有一个 Hexo 项目；
- 想稳定输出技术博客；
- 希望把“写作 -> 预览 -> 发布 -> 推送源码”流程标准化。

## 项目结构与部署机制（先理解）

你的博客目录核心是：

```text
hexo_Blog/
├── _config.yml
├── package.json
├── source/_posts/
├── themes/hexo-theme-fluid/
├── public/
└── .deploy_git/
```

你当前采用的是“同仓库双分支”模式：

- `hexo` 分支：存源码（Markdown、配置、主题改动）；
- `main` 分支：存静态站点文件（由 `hexo g && hexo d` 生成并推送）。

只要记住一句话：**平时改源码在 `hexo`，线上页面来自 `main`。**

## 环境准备（第一次或新机器）

### 1. 进入项目并安装依赖

```bash
cd /Users/coisini/Documents/Projects/Blog/hexo_Blog
npm install
```

预期结果：

- 生成/更新 `node_modules`；
- 无致命报错（warning 可后续处理）。

### 2. 确认关键脚本可用

```bash
npm run clean
npm run build
npm run server
```

预期结果：

- `build` 后看到类似 `INFO  Files loaded in ...`；
- `server` 后访问 `http://localhost:4000` 能打开站点。

## 日常写作流程（推荐固定动作）

下面这套是我最推荐的“每天可重复流程”。

### Step 1：创建文章草稿

```bash
npx hexo new post "你的文章标题"
```

新文件会出现在：

```text
source/_posts/你的文章标题.md
```

你的配置里 `post_asset_folder: true`，通常也会生成同名资源目录。你可以把该文章图片放进去，避免图片散落。

### Step 2：填写 Front Matter

建议至少填这些字段：

```yaml
---
title: 文章标题
date: 2026-03-07 12:00:00
updated: 2026-03-07 12:00:00
tags: [主题A, 主题B]
categories: [技术分享]
---
```

实操建议：

- `tags` 控制检索粒度；
- `categories` 保持稳定层级，不要太碎；
- `updated` 每次大改都更新。

### Step 3：本地预览边写边看

```bash
npm run server
```

访问：`http://localhost:4000`

建议检查：

- 标题和摘要是否自然；
- 代码块高亮是否正常；
- 图片路径是否可访问；
- 移动端排版是否可读。

### Step 4：发布前本地生成

```bash
npm run clean
npm run build
```

为什么必须这一步：

- `clean` 防止旧缓存导致“本地好、线上错”；
- `build` 提前发现渲染错误（比如 front matter、模板、路径问题）。

### Step 5：部署到 GitHub Pages

```bash
npm run deploy
```

这一步会把生成后的静态文件推到你配置的仓库分支（`main`）。

### Step 6：推送源码分支（很重要）

```bash
git add .
git commit -m "feat: 发布<文章标题>"
git push origin hexo
```

不要只部署不推源码。否则你本地机器坏了，`hexo` 分支没同步，后续很难维护。

## 一套可复制的“发布日”命令清单

每次发布前后，按下面执行基本不会漏：

```bash
cd /Users/coisini/Documents/Projects/Blog/hexo_Blog
git checkout hexo
git pull origin hexo

# 写作/改稿完成后
npm run clean
npm run build
npm run deploy

git add .
git commit -m "feat: 发布新文章"
git push origin hexo
```

## 常见问题与排查

## 1) 部署成功但线上没变化

先检查：

- 是否真的执行了 `npm run build` 和 `npm run deploy`；
- `_config.yml` 的 `deploy.repo` 和 `branch` 是否正确；
- GitHub Pages 是否指向 `main`（仓库设置里确认）。

## 2) 本地能看，线上样式错乱

优先排查：

- `url` 是否正确（你当前是 `https://zy-pioneer.github.io`）；
- 是否有混合协议资源（HTTP/HTTPS）；
- 主题配置是否改动后未重新 `clean`。

## 3) 分类/标签页面突然少了

常见原因：

- 文章 front matter 丢失 `categories` / `tags`；
- 清理后未重新完整生成；
- 某篇文章格式错误导致部分索引未生成。

## 4) 图片不显示

确认三件事：

- 图片是否放在对应文章资源目录；
- Markdown 引用路径是否匹配生成后的目录；
- 文件名大小写是否一致（部署环境区分大小写）。

## 5) `.deploy_git` 里出现大量删除

这通常不是灾难，常见于你做了 `clean` 后重新生成。处理原则：

- 先重新 `build`，确认目标页面都重新产出；
- 再 `deploy`，让 `main` 分支与最新 `public` 同步。

## 发布质量检查（建议固定为 Checklist）

发布前用 1 分钟过一遍：

- 标题是否清晰包含关键词；
- 开头 3 段是否讲清“读者收益”；
- 代码/命令是否可复制运行；
- 是否给出“下一步”建议；
- 本地页面与移动端都检查过。

## 结语

把博客运营稳定下来，不靠记忆，靠流程。  
你只要每次都执行“写作 -> 预览 -> clean/build -> deploy -> push hexo”，基本就能长期稳定维护。

如果你愿意，我下一篇可以继续写一篇配套教程：  
`Hexo 图片管理与文章资源目录最佳实践（post_asset_folder 实战）`。
