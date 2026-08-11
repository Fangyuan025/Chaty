---
name: tiktok-video
description: 一句话需求 → 成品竖屏短视频:写分镜、自动搜免费素材、TTS 配音、逐字卡拉OK字幕、BGM、ffmpeg 合成 1080x1920 MP4(零 API key 可用)
when: the user asks to create / make / generate a short video(做短视频/抖音视频/TikTok/Shorts/Reels)on any topic
---

# TikTok / 抖音短视频生成

你做**创意**部分——文案、分镜、搜索关键词、质量审查;`{SKILL_ROOT}/scripts/` 里的脚本做**机械**部分——TTS 逐字时间戳、素材搜索下载、字幕渲染、ffmpeg 合成、响度标准化。脚本已随 Chaty 就位,勿重写。

前置:`ffmpeg` + `python3` 在 PATH(macOS: `brew install ffmpeg`;Windows 需 Git Bash,且下文的 `.venv/bin/python` 换成 `.venv/Scripts/python`)。零 API key 即可跑;环境变量 `PEXELS_API_KEY` / `PIXABAY_API_KEY`(免费申请)可解锁实拍视频片段,有则优先。

## 0. 一次性安装(若 `{SKILL_ROOT}/.venv` 已存在则跳过)

```bash
bash {SKILL_ROOT}/scripts/setup.sh
```

## 1. 先核实,再写文案 + 分镜

**语言规则(最先决定)**:视频语言 = **用户需求的语言**(或用户明确指定的目标受众),永远不是本文档或示例的语言。英文需求 → `lang: "en"`,文案、hook、角标("No.1"/"TOP 1")、CTA 全英文,en-US 音色,TikTok 习惯;中文需求 → `lang: "zh"`,抖音习惯("第1名")。用户点名目标市场("给美国观众")则以市场为准。`lang` 缺省管线会直接报错,没有默认值。

**模型的内部知识可能过期或有错——未经核实的论断禁止进视频。**动笔前用你的联网搜索工具核实文案里每一个具体论断:数字、统计、纪录、价格、日期、排名、一切"第一/最大/唯一/最快",以及任何时效性内容(新闻/产品/版本/"今年/最新"),无论你多有把握。规则:**按来源改写文案**(不是反过来);核实不了就换成能核实的或直接删;把依据的 URL 记入 storyboard 的 `"sources": [...]`(会出现在 report.txt 供用户复核);确实无联网能力时只用教科书级常识、避免具体数字、并向用户声明未核实。

然后读 `{SKILL_ROOT}/references/writing-guide.md`(hook 公式、节奏、场景结构)。然后在工作区建 `<slug>/storyboard.json`:

```json
{
  "title": "深海里最诡异的3种生物",
  "lang": "zh",
  "aspect": "9:16",
  "voice": "zh-CN-YunjianNeural",
  "rate": "+10%",
  "caption_style": "karaoke",
  "bgm": {"mood": "mystery", "gain_db": -16},
  "hook": {"text": "深海禁区", "seconds": 2.3},
  "scenes": [
    {
      "text": "你知道吗?在阳光永远照不到的深海,藏着比科幻电影更诡异的生物。",
      "keywords": ["deep sea NOAA ocean exploration", "submarine dark ocean"],
      "providers": ["openverse", "wikimedia"],
      "effect": "kb_in"
    },
    {
      "text": "第一名,鮟鱇鱼。头顶挂着一盏会发光的钓鱼灯,守在黑暗里等猎物送上门。",
      "keywords": ["humpback anglerfish", "anglerfish museum"],
      "badge": "第1名"
    }
  ]
}
```

字段速查:

| 字段 | 取值 | 说明 |
|---|---|---|
| `lang` | `zh` \| `en`(**必填**) | = 用户需求/受众的语言(见上方语言规则);决定字幕分组和默认音色 |
| `aspect` | `9:16`(默认) \| `16:9` \| `1:1` | |
| `voice` | 任意 edge-tts 音色 | zh: `zh-CN-YunjianNeural`(磁性男) `zh-CN-XiaoxiaoNeural`(女) `zh-CN-YunxiNeural`(阳光男);en: `en-US-ChristopherNeural` `en-US-AriaNeural` |
| `rate` | 如 `+10%` | 营销号节奏:zh `+8%`~`+15%`,en `+5%`~`+10%` |
| `caption_style` | `karaoke` \| `pop` \| `none` | karaoke = 逐字高亮(推荐) |
| `bgm` | `{"query":"风格词"}` \| `{"mood":…}` \| `{"file":"路径"}` \| `{"mood":"none"}` | **默认选 `mood`**:曲表(upbeat funny inspiring chill tech mystery epic sad horror)是无数营销号在用的 MacLeod 熟脸配乐,按项目随机换曲;**表内没有贴合风格的才用 `query` 搜**(ccMixter/Jamendo CC 曲,节拍器自动筛掉没鼓点的):盘点/悬念 `"trap"`、种草 `"lofi chill"`、励志 `"epic cinematic"`、搞笑 `"quirky"`;两者可同设(query 先试、mood 兜底);热门歌用 `file`(版权自负)|
| `beat_sync` | `true`(默认) \| `false` | BGM 自动节拍分析,所有切镜吸附到节拍上(卡点);人声永不截断,只伸缩场景尾部留白;鼓点弱的曲子会自动跳过 |
| `bgm.vibe` | `"spedup"` \| `"slowed"` \| 不设 | 抖音标志性音色:spedup ≈1.25×提速升调(卡点/盘点主流),slowed = 减速+混响(情感向);任何 bgm 来源都可加,处理后自动重测节拍 |
| `hook` | `{"text","seconds"}` | 开头大字标题卡,≤8 字/词 |
| `sticky_title` | `{"text":…}` | 可选的顶部常驻话题条;默认关,用户要才加 |
| `sfx` | `true`(默认) \| `false` | 转场 whoosh 音效 |
| 场景 `keywords` | **英文、具体名词**的列表 | **每条 = 一个镜头画面**;视频约每 3s 切一镜,场景超过 ~4s 就给 2–3 条(给少了会循环补拍) |
| 场景 `badge` | 如 `"第1名"` / `"TOP 1"` | 场景开头的大号盖章标签,榜单类必用 |
| 场景 `providers` | 列表 | 免 key 图片:`openverse` `wikimedia` `nasa`;**免 key 实拍视频**:`wikimedia_video`(Commons 视频转码)`nasa_video`(PD 太空/科学)`archive_video`(Prelinger 历史胶片);有 key:`pexels_video` `pexels_photo` `pixabay_video` |
| 场景 `effect` | `auto` `kb_in` `kb_out` `pan_left` `pan_right` `static` | 第一镜的 Ken Burns 动效,后续镜头自动轮换 |
| 场景 `media` | 文件路径 | 跳过搜索,用你自己准备的素材 |

**经验值(发布线):总长 45–75s、6–9 个场景、每场景一个观点、2–3 个镜头。**文案中文 20–45 字 / 英文 14–30 词。低于 ~40s 或全程单镜头会显单薄,别交付。

## 2. 跑管线

```bash
{SKILL_ROOT}/.venv/bin/python {SKILL_ROOT}/scripts/pipeline.py <slug>
```

也可分段:`tts.py` → `assets.py` → `bgm.py` → `compose.py` → `check.py`(都接 `<slug>` 参数)。改过 storyboard 后重合成很快:`pipeline.py <slug> --skip-tts --skip-assets`;但改了场景**文案**必须重跑 `tts.py`。

## 3. 审查——质量分水岭(必做)

免费图片搜索不完美,**像人类编辑一样审**:

1. assets 阶段后用 `view_image` 看 `<slug>/media/assets_sheet.jpg`。镜头标号 `01a 01b 02a…`,任何一镜与文案不符就换词重搜:

   ```bash
   {SKILL_ROOT}/.venv/bin/python {SKILL_ROOT}/scripts/assets.py <slug> --scene 3 --shot 2 --keywords "better english nouns"
   ```

   (重搜自动拉黑被否素材,重复到全部匹配;也可自己下载图片后把场景 `media` 指向它。)**目检确认**某个被 `[!]` 标记的镜头其实没问题时(如无描述标题的好图),在 `media/manifest.json` 给该 shot 加 `"approved": true` 消除误报——眼见优先于文本审计,严禁盲批。

2. compose+check 后用 `view_image` 看 `<slug>/review/contact_sheet.jpg`:字幕可读且同步、无错帧丑帧、hook 可见。有问题就改 storyboard 重合成,直到像营销号编辑敢发的成片。

`assets.py` 和 `check.py` 都会自动比对每格素材的来源标题和场景关键词,跑题的直接打 `[!] OFF-TOPIC` 并给出重搜命令(assets 阶段就 flag,别等合成完)——**任何 `[!]` 都必须处理完才能交付**。若你无法真正看到图片(非视觉模型),这些标题 flag 就是你的审查:重搜 → `--skip-tts --skip-assets` 重合成 → 重跑 check,直到零 `[!]`。

`review/report.txt` 含时长/响度检查和**素材署名文本**(CC-BY 来源 + 音乐 credit)——连同 `final.mp4` 一并交给用户。

## 搜索技巧(最大质量杠杆)

- 关键词必须**英文**、1–4 词、**具体可见的名词**("humpback anglerfish"、"scuba diver silhouette"),忌抽象概念("mystery"、"success")。
- `wikimedia` 擅长动物/科学/历史/地标;`openverse`(Flickr 等)擅长生活方式/风景/氛围;`nasa` 管太空;有 key 优先 `pexels_video`/`pixabay_video` 实拍。
- 加语境词消歧:"NOAA"、"museum"、"aquarium"、"macro"。
- 某场景反复搜不到就换**视觉概念**而不只是换词(讲"仅5%被探索"→拍剪影潜水员,别搜"statistics")。
- `wikimedia_video` 支持负词过滤——气象/太空搜索会被卫星云图淹没,写 `"lightning storm -CIRA -satellite -JPSS"`;非英语词能挖到更多实拍("Blitz Gewitter"、"tormenta rayos");优先具体名词("Fagradalsfjall lava drone");视频缩略图务必目检:提防内嵌字幕、水印、软件教程和 CGI 宣传片。
- **中国街景配方**:Commons 有整个城市漫步视频体裁——搜 `"Walking China 城市名"`、`"Riding a street in Shanghai"` 或具体地点(`"Huaqiangbei electronics market"`),分类检索 `incategory:"Videos of Shanghai"` 也可用;这些素材都是英文标题,中文查询几乎搜不到。

## 排障

- edge-tts 网络错误:自带 4 次重试,仍失败就重跑 `tts.py`。
- BGM 下载失败:重跑 `bgm.py`(有缓存),或 `"mood":"none"`,或放一个 mp3 到 `{SKILL_ROOT}/assets/bgm/` 并用 `bgm.file` 指向它。
- 单个 provider 报错/空结果无妨,其余会补位;看每场景日志。
- 字幕 emoji 仅 macOS 可渲染,别用 emoji 承载语义。

## 交付

给用户:`<slug>/final.mp4` + `report.txt` 里的署名文本;若要封面给 `<slug>/review/cover.jpg`。
