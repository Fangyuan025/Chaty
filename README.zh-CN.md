<div align="center">

[English](README.md) · **简体中文** · [Português (BR)](README.pt-BR.md)

<img src="icon.png" width="88" height="88" alt="Chaty" />

# Chaty

### 私密的本地 AI —— 你的模型、你的数据、你自己的设备。

Chaty 是一款精致的桌面应用,让开源大模型**完全离线**运行。
无需账号、不上云、零遥测 —— 还内置本地编码智能体、文档知识库、
Deep Research 与免手语音。

[![Latest release](https://img.shields.io/github/v/release/Fangyuan025/Chaty?label=release&color=19c37d)](../../releases/latest)
[![Downloads](https://img.shields.io/github/downloads/Fangyuan025/Chaty/total?color=8a63d2)](../../releases)
[![CI](https://img.shields.io/github/actions/workflow/status/Fangyuan025/Chaty/ci.yml?branch=main&label=CI)](../../actions)
[![Windows · Vulkan](https://img.shields.io/badge/Windows-Vulkan-0078D6?logo=windows&logoColor=white)](../../releases)
[![macOS · Metal + MLX](https://img.shields.io/badge/macOS-Metal_%2B_MLX-000000?logo=apple&logoColor=white)](../../releases)
[![100% offline](https://img.shields.io/badge/100%25-offline-19c37d)](https://chaty.ca)
[![Rust + Tauri 2](https://img.shields.io/badge/Rust_+_Tauri_2-CE412B?logo=rust&logoColor=white)](#架构)
[![License: MIT](https://img.shields.io/badge/License-MIT-444)](LICENSE)

[**↓ 下载**](../../releases) · [**官网**](https://chaty.ca) · [**文档**](https://chaty.ca/docs.html) · [**Hugging Face 上的 Chaty 模型**](https://huggingface.co/stevenpr/chaty-qwen3.5-4b-design-GGUF)

<br />

<img src="docs/screenshots/demo.gif" width="860" alt="Chaty 本地编码智能体:一键授权后读取工作区外文件并总结" />

<sub>一个本地编码智能体 —— 搜 GitHub、读源码、改你的文件、跑测试。**全在你自己的机器上。**</sub>

</div>

---

## 为什么选 Chaty

- 🔒 **真正私密** —— 每个模型、文档、对话都留在你的设备上。无需注册、没有服务器、不向任何地方回传。
- ⚡ **原生而快** —— Rust + llama.cpp 内核,**Vulkan / Metal** GPU 卸载,按硬件自动调优,放不下时平稳回退 CPU。
- 🧰 **不只是聊天框** —— 编码智能体、知识库(RAG)、Deep Research、免手语音,以及会自愈的设计画布 —— 全部离线。
- 🧠 **几乎什么都能跑** —— Llama 3、Gemma 3 / 4、Qwen 3 / 3.5 / 3.6、Hugging Face 上的*任意* GGUF,**Apple Silicon 上还能原生跑 MLX 模型** —— 还有 **Chaty 自研微调模型**。
- 💻 **对弱硬件友好** —— 首次启动的「为我配置」会按你的内存挑一个合适的模型,一键下载。

<br />

## 一个本地编码智能体

拨动 **Chat · Code** 开关,Chaty 就成了你代码库的智能体。选一个文件夹、描述任务,
它便自主探索、修改并验证项目 —— 每一步实时可见,每处改动都先审批 + 看 diff。

- 🌐 **整个互联网都是它的工具** —— 无 key 站内搜索 **GitHub**(仓库、issue、*代码*)、Reddit、YouTube、B站及任意域名;抓取按内容自适应(文章→Markdown、PDF→文本、视频→字幕转写)。
- 🧭 **能开真实浏览器** —— 打开网页、把动态内容当文字读、用真实鼠标事件点击和整表单填写、登录、翻页 —— 该用视觉时才截图亲眼看。
- 🧠 **会思考的工具** —— `understand_repo` 一次摸清仓库、`search_code` 按相关度排序、`read_file` 只取一个符号加全部调用处、`validate_change` 只跑与改动相关的测试。粗活下沉进工具,小模型只做决策。
- ✏️ **精确编辑 + 真 shell** —— 带 diff 预览与**语法门**的精确文本补丁,以及沙箱限定在工作区内的命令与长时**后台任务**(dev server、构建)。
- ⏪ **一切由你掌控** —— 逐条审批、命令白名单、对读到的一切内容做防注入,以及**检查点一键回滚**:恢复文件*并*回退对话。
- 🔌 **为小模型定制的 MCP** —— 连接任意 Model Context Protocol 服务器(stdio 或 Streamable HTTP),或一键添加**版本钉死、经真连认证的精选商店条目**。工具文档自动瘦身,16K 上下文也装得下任意多服务器;所有结果过防注入,未信任服务器逐次审批。
- 📚 **技能与项目记忆** —— 把一页步骤写成 `SKILL.md` 放进 `~/.chaty/skills/`(或项目级),智能体只在相关时加载;`remember` 把非显而易见的发现存进 `.chaty/memory/`,下一个会话开局即知。纯 Markdown、人可编辑、永不离开本机。

<details>
<summary>更多 Code 模式细节</summary>

- 直接读 **PDF / Word / Excel / PowerPoint**(扫描件自动 OCR);`search_files` 按名字或内容查找;outline 大纲导航大文件;补丁失配时给「你是不是想改这里」提示。
- 浏览器自动化在真实网站上端到端验证过,还能开进你的真实 Chrome —— 全程围观,登录状态也保留。
- 为本地模型而生:**Off / Normal / Deep** 思考强度开关、**提示词处理进度环**、上下文用量环 + 自动压缩、按上下文窗口定预算的整文件读取、`search_code` 语义检索 + 知识库 `search_docs`,以及防复读循环打断。
- 会话持久化、项目记忆(**AGENTS.md**)、自定义 **/技能** 与 slash 命令。
- 在**设置 → Code** 里调:单轮步数上限、命令超时、步骤温度、自动批准编辑开关、后台运行浏览器开关,以及命令白名单。
- 文件访问永远不出你选的文件夹;工作区外访问按目录询问;sudo 会先询问并有安全的密码输入框;下载落进工作区,也一并纳入检查点回滚。

</details>

<br />

## 跑分

下表每一行都是**同一个本地模型**——Qwen3.5-35B-A3B(MoE,每 token 仅 ~3B 激活),mxfp8 · MLX,关闭思考,全程单机:

| SWE-bench Verified — 45 题 macOS 验证子集 | 解出 |
| --- | --- |
| **Chaty 智能体(v1.9)**——完整工具链,16K 上下文 | **15/45(33%)** |
| qwen-code 0.20——模型家族官方 CLI(需 32K) | 12/45(27%) |
| pi 0.81——极简四工具 agent CLI | 10/45(22%) |
| opencode 1.18 | 7/45(16%) |
| 裸 bash 智能体——单工具消融对照 | 6/45(13%) |

同模型、同任务、同判分、同一台机器——五种智能体设计同台。Chaty 领跑全场:领先模型家族自家的官方 CLI([qwen-code](https://github.com/QwenLM/qwen-code))且**只用它一半的上下文窗口**,是裸 bash 消融的 **2.5 倍**。这正是设计论点的实测版:前沿大模型配一层薄脚手架就够用,而**小模型上,智能必须下沉到工具里**——仓库感知检索、符号级阅读、精确编辑、恢复护栏、编辑后诊断。方法学、各家配置与诚实对比说明(子集、macOS 环境——**不可**与官方排行榜数字直接对比):[docs/BENCHMARKS.md](docs/BENCHMARKS.md)。

<br />

## 设计画布

<picture>
  <source media="(prefers-color-scheme: light)" srcset="docs/screenshots/canvas-hero-light.jpg" />
  <img src="docs/screenshots/canvas-hero-dark.jpg" width="860" alt="设计画布:实时预览与真实源码并排,元素↔代码行对照,控制台" />
</picture>

- **预览 | 代码,并排呈现** —— 每个页面都在分栏工作室中打开:左边实时预览,右边**真实源码**,语法高亮、跟随你的代码配色。三栏宽度自由拖拽,支持全屏、页面刷新,以及镜像页面日志与报错的**控制台**标签。
- **指哪改哪** —— 对照模式把两栏双向连起来:悬停元素,代码跳到对应行;点代码行,页面元素闪烁定位。**点击即选中**(⌘/Ctrl 多选),下一条指令只改选中的元素 —— 想亲手改就点**编辑**按钮直接开源码。
- **亲眼看着它改** —— 迭代过程 Cursor 式流式呈现:代码栏逐行扫描全文,完成后落到**变更**视图(+N/−N,与 Code 模式同款红绿 diff)。

<picture>
  <source media="(prefers-color-scheme: light)" srcset="docs/screenshots/canvas-scan-light.jpg" />
  <img src="docs/screenshots/canvas-scan-dark.jpg" width="860" alt="模型修改页面时的逐行扫描" />
</picture>

- **自愈修复,版本留存** —— 运行时错误给出一键**修复**(始终先征求同意);兼容层保证真浏览器里能跑的页面在画布里同样干净(history 路由、cookie、剪贴板);每条回答的画布会话关闭再开都在,版本历史可回退、可确认重置,并可导出为独立 `.html`。

<br />

## 什么都能渲染的聊天

<table>
<tr>
<td width="50%"><img src="docs/screenshots/shot-chat.jpg" alt="富文本渲染:语法高亮代码、表格与 KaTeX 数学" /></td>
<td width="50%"><img src="docs/screenshots/shot-chat-light.jpg" alt="同一段对话在 Chaty 浅色主题下" /></td>
</tr>
</table>

- 流式、可折叠的 **`<think>`** 面板,随生成自动跟随模型推理。
- **KaTeX** 数学、表格、**Mermaid** 图、逐块代码复制,以及应用内渲染单文件 HTML —— 含可玩的网页游戏。
- **⌘K 命令面板**、可置顶/重命名的对话、拖拽附件、导出(Markdown / JSON)与全文搜索。
- 四套配色(深浅各两款,可跟随系统)、原生界面缩放、减少动态效果支持,以及 **English / 简体中文 / Português (BR)** 界面。

<br />

## Chaty 能看见

加载一个**视觉模型**(权重与 `mmproj` 编码器放在同一个文件夹里,自动配对),识图能力就会在各处打开:

- **聊天** —— 附一张图直接问;追问也很快(看过的图不再重复编码)。
- **Code** —— 智能体能读截图、用 `view_image` 看任意图片;输入框像聊天一样收图片和文档。
- **知识库** —— 导入的图片除 OCR 外还会生成一段文字描述,让你能搜到图里*画的是什么*。
- **画布** —— 让它改页面时,模型能看到当前渲染出的实际效果。

纯文本模型继续走 OCR,不影响任何原有功能 —— 而从旧版本升级时,一个一次性弹窗会引导你把散放的 `.gguf` 一键归入「一个模型一个文件夹」的布局。

<br />

## 模型:商店、原生 MLX,以及 Chaty 自研

- 内置**模型商店**:按名称或作者搜索 Hugging Face,按 **GGUF / MLX** 筛选、按热门/下载量排序 —— 从下拉框选一个**量化版本**直接下载。看到的是模型,不是文件列表。
- 参数量 / 架构 / 视觉徽章、应用内直接渲染仓库 README,还有按你机器内存给出的**「可完整载入内存」**提示。视觉模型自动附带编码器;粘贴仓库链接的老用法依然可用。
- **MLX 原生运行**(Apple Silicon):mlx-community 的文件夹模型通过 Apple MLX 栈在独立侧车进程中运行 —— 对话、视觉、思考开关、Code 智能体、知识库与 GGUF 完全同级,弹出模型时内存*必定*全数归还。
- **Chaty 自研微调** —— 从更大的教师模型蒸馏而来的 Qwen3.5-4B,为本地单文件网页设计调校,内置 Chaty 身份认同与带引用的回答。「为我配置」里的一键选项,在 **[Hugging Face](https://huggingface.co/stevenpr/chaty-qwen3.5-4b-design-GGUF)** 完全开源。

<br />

## 一个私密的知识库

<table>
<tr>
<td width="52%">

- 把 **PDF、Word、Excel、Markdown、约 90 种文本/代码格式、图片**索引进本地库 —— 单文件或整文件夹。图片会走 **OCR**,配合视觉模型还会**用文字描述画面内容**,让你能搜到图里画的是什么。
- **混合检索**:bge-m3 向量 + BM25 关键词,RRF 融合、MMR 去重、邻接分块扩展。
- **严格 grounding** —— 答案只来自你的文档,**按文件引用**并悬停预览出处段落;没覆盖到的内容 Chaty 会直说,而不是瞎猜。
- **一键报告** —— 对整个知识库生成带引用的 NotebookLM 式综述,可导出 PDF / Markdown。

</td>
<td width="48%"><img src="docs/screenshots/shot-knowledge.jpg" alt="本地知识库:带逐文件开关的索引文档,以及一键报告 / 播客" /></td>
</tr>
</table>

<br />

## Deep Research 与联网

- 给一个主题,Chaty 会规划查询、进行**多轮**联网搜索并穿插推理,最终写出结构化的带引用报告 —— **可导出 PDF 或 Markdown**。
- 天然诚实:参考文献只列它真正引用过的来源。
- 免费、免密钥的多 provider 搜索链(Brave → Bing → DuckDuckGo → Wikipedia),单个被封不会让搜索失效。

<br />

## 免手语音

<table>
<tr>
<td width="48%"><img src="docs/screenshots/shot-live.jpg" alt="实时语音模式:一个动态光球,连续免手对话" /></td>
<td width="52%">

- **实时模式** —— 配一个动态光球的连续、免手语音对话。
- 语音输入/输出,静音自动发送 + 朗读 —— **11 种嗓音** + 语速调节。
- **深读播客** —— 把知识库变成 NotebookLM 风格的双主持人音频节目,支持 WAV 导出。
- 所有语音都跑在 **CPU** 上,绝不与大模型抢显存。

</td>
</tr>
</table>

<br />

## 一切都留在你的机器上

<table>
<tr>
<td width="52%">

- 会话、模型、索引都在一个**本地数据文件夹**里 —— 拷走即备份,一键即清空。
- **GPU 加速**:跨厂商 **Vulkan**(Windows)与 **Metal**(Apple Silicon,统一内存下全量卸载),按显存自动调优,带 OOM 回退与 CPU 兜底。
- **任意 `.gguf` 或 MLX 文件夹** —— 分词器与对话模板都取自文件本身;一流支持 Llama 3、Gemma 3 / 4 与 Qwen 3 / 3.5 / 3.6。
- **可调上下文**,自动把模型训练长度适配到你的内存,接近上限时总结较早的对话;**安全切换模型**,完整采样控制 + 可保存预设。

</td>
<td width="48%"><img src="docs/screenshots/shot-settings.jpg" alt="设置:展示会话、模型与知识库统计的本地数据面板" /></td>
</tr>
</table>

> **离线优先。** 网络仅用于可选的联网搜索和一次性模型下载。

<br />

## 安装

到 [**Releases**](../../releases) 页面获取最新构建:

| 平台 | 文件 | 说明 |
|---|---|---|
| Windows x64 | `Chaty_*_x64-setup.exe` | 单用户安装,无需管理员权限 |
| macOS（Apple Silicon） | `Chaty_*_aarch64.dmg` | 见下方首次启动说明 |

**macOS 首次启动。** Chaty 是 ad-hoc 签名但未公证(背后没有付费的 Apple Developer 账号),
所以首次打开时 Gatekeeper 会报警。应用是安全的 —— 一切都在本地运行。清除一次下载隔离属性即可:

```sh
xattr -dr com.apple.quarantine /Applications/Chaty.app
```

然后正常打开 Chaty。(或:先打开、忽略警告,再到 **系统设置 → 隐私与安全性 → 仍要打开**。)
macOS 上可写的模型文件夹在应用数据目录里 —— 用模型菜单里的 **打开模型文件夹**。

## 构建

详见 **[BUILD.md](BUILD.md)**。

```powershell
# Windows
npm install
.\dev.ps1                            # 开发
npm run tauri build -- --no-bundle   # 生成 exe → 再编译 Inno 安装包
```

```bash
# macOS（Apple Silicon）
npm install
npm run tauri dev      # 开发(Metal)
npm run tauri build    # → .app + .dmg
```

发布由 CI 完成:用 `scripts/bump-version.sh x.y.z` 升版本,推一个 `vx.y.z` tag —— GitHub Actions 会把两个平台的安装包打到同一个 release。

## 架构

| 层 | 技术栈 |
|---|---|
| 外壳 | Tauri 2 —— 系统托盘、全局快捷键、单实例 |
| 前端 | React 19 · Vite · react-markdown · KaTeX |
| 推理 | Rust · `llama-cpp-2`(llama.cpp)—— Vulkan(Windows)/ Metal(macOS) |
| 语音 | `sherpa-rs`(ONNX Runtime,CPU)—— Whisper-base.en + Kokoro-82M |
| 知识库 | bge-m3 向量 + BM25 · 混合 RRF / MMR 检索 · SQLite 向量库 |
| 存储 | SQLite —— 会话、消息、全文搜索 |

## 许可证

MIT —— 见 [LICENSE](LICENSE)。基于 [llama.cpp](https://github.com/ggml-org/llama.cpp)、[Tauri](https://tauri.app) 与 [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) 构建。
