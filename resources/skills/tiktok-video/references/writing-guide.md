# 营销号级短视频文案指南 / Marketing-grade Short-Video Script Guide

The storyboard you write IS the video. Spend real effort here.

## 结构公式 / Structure formula

**钩子 Hook (scene 1, 0–4s) → 信息点/冲突 Body (3–5 scenes) → 反转/升华 Payoff → 关注引导 CTA (last scene)**

Total 30–60s. One idea per scene. Never explain — assert.

## 钩子公式 / Hook formulas (scene 1 text + `hook` title)

| 类型 | 中文示例 | English example |
|---|---|---|
| 悬念 curiosity gap | 你知道吗?在阳光永远照不到的深海… | Did you know there's a place sunlight has never touched? |
| 反常识 counter-intuitive | 越贵的防晒霜,可能越没用。 | The most expensive sunscreen might be the most useless. |
| 数字清单 listicle | 3个让你效率翻倍的AI工具,最后一个99%的人不知道。 | 3 AI tools that double your output — #3 is almost unknown. |
| 恐惧/损失 fear of loss | 这个习惯,正在悄悄毁掉你的睡眠。 | This one habit is quietly ruining your sleep. |
| 挑战权威 challenge | 老师绝对不会告诉你的3个历史真相。 | 3 history facts your teacher never told you. |

`hook` title: ≤8 chars (zh) / ≤5 words (en), punchy noun phrase:
"深海禁区", "睡眠杀手", "SLEEP KILLER", "HIDDEN HISTORY".

## 文案节奏 / Pacing rules

- **总时长 45–75 秒,6–9 个场景**。短于 40 秒的视频信息密度不够,不要交付。
- 短句。一句一个信息点。删掉所有"的话/其实/那么/basically/actually"。
- zh: 每场景 20–45 字;en: 14–30 words per scene.
- 清单类用 "第一名/第二名" "Number one" — 并给对应场景加 `badge`("第1名"/"TOP 1")。
- 每个场景给 **2–3 个 keywords**(= 2–3 个镜头),画面每 ~3 秒切换一次;
  只给 1 个 keyword 的长场景会显得单调。
- 每个场景结尾留半拍悬念,把观众推向下一句。
- CTA 固定收尾: "关注我,带你看更多XX" / "Follow for more XX you won't believe."

## 语音 / Voice choices

- zh 悬念/科普/硬核: `zh-CN-YunjianNeural` rate +10%
- zh 生活/情感/种草: `zh-CN-XiaoxiaoNeural` rate +8%
- zh 轻松/搞笑: `zh-CN-YunxiNeural` rate +12%
- en authoritative/docu: `en-US-ChristopherNeural` +5%
- en energetic/listicle: `en-US-AriaNeural` +8%,  deep hype: `en-US-GuyNeural`

## BGM 选择

**默认直接从 `mood` 表选**(upbeat funny inspiring chill tech mystery epic sad
horror)——这批 Kevin MacLeod 曲子正是无数营销号/faceless 频道在用的"熟脸"配乐,
观众听着就是"这类视频该有的声音";同一 mood 会按项目随机换曲,不会千篇一律。

**mood 表里没有贴合内容气质的,再用 `bgm.query` 去搜**(ccMixter/Jamendo 数千首
CC 曲,自动用节拍器筛掉没鼓点的);配合默认开启的 `beat_sync`,切镜自动卡拍:

| 内容类型 | mood 首选 | 表内不合适时 query | vibe 建议 |
|---|---|---|---|
| 盘点/悬念/猎奇 | mystery | "trap" / "phonk" | `"spedup"`(节奏更催) |
| 种草/生活/vlog | chill / upbeat | "lofi chill" / "hip hop" | 不加或 `"slowed"` |
| 励志/震撼/史诗 | epic / inspiring | "epic cinematic" | 不加 |
| 科技/数码 | tech | "electronic dance" | 可加 `"spedup"` |
| 搞笑/整活 | funny | "quirky" | 不加 |
| 情感/怀旧 | sad / chill | "chill" / "hip hop" | `"slowed"`(slowed+reverb) |

用户指定的热门歌用 `bgm.file`(版权自负)。

## 视觉 / Visual planning per scene

While writing each scene, already decide **what the viewer sees**. Write
keywords for a *specific photographable subject*:

- ❌ "success mindset" → ✅ "sunrise mountain climber silhouette"
- ❌ "AI technology" → ✅ "server room blue lights" / "robot arm factory"
- Alternate effects: kb_in → kb_out → pan; use `static` for text-heavy art.
- Illustration/vintage-poster images are fine and even stylish for
  history/science topics (wikimedia has many).

## 质量红线 / Quality bar (reject your own draft if…)

- **任何数字/纪录/"最"字论断没有经过联网核实**(模型记忆会过期、会错;
  按来源改写文案,核实不了就删,来源 URL 记入 `sources`)
- Hook 前 3 秒没有制造"必须看下去"的理由
- 任何场景的画面与文案无关(观众 0.5 秒就会划走)
- 总时长 > 70s 或 < 20s;字幕一行超过 10 个汉字
- 结尾没有 CTA
