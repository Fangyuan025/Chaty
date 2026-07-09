// Built-in Code-mode skills: curated task templates, invoked as /name from the
// composer. Users can disable any of them (Settings → Code) and add their own
// custom skills alongside; a custom skill with the same name shadows a builtin.

export interface BuiltinSkill {
  name: string;
  desc: { zh: string; en: string };
  prompt: { zh: string; en: string };
}

export const BUILTIN_SKILLS: BuiltinSkill[] = [
  {
    name: "init",
    desc: { zh: "分析项目并生成 PROJECT.md 导览", en: "Analyze the project and write PROJECT.md" },
    prompt: {
      zh: "分析这个项目:目录结构、技术栈、入口文件、构建/测试方式和关键模块,然后在根目录写一个 PROJECT.md,作为之后编码任务的导览(简洁、面向开发者)。",
      en: "Analyze this project — layout, stack, entry points, build/test commands, key modules — then write a concise developer-oriented PROJECT.md at the root to guide future coding tasks.",
    },
  },
  {
    name: "review",
    desc: { zh: "代码审查:找 bug 与坏味道(不改代码)", en: "Code review: bugs & smells (read-only)" },
    prompt: {
      zh: "对这个项目做一次代码审查:找出潜在 bug、边界情况问题和明显坏味道,按严重程度输出问题清单(文件:行号 + 一句话说明 + 建议)。只读分析,不要修改任何文件。",
      en: "Review this codebase: find likely bugs, edge-case issues, and clear code smells. Output a severity-ordered list (file:line + one-line issue + suggestion). Read-only — do not modify any files.",
    },
  },
  {
    name: "test",
    desc: { zh: "为代码补测试并跑通", en: "Add tests and make them pass" },
    prompt: {
      zh: "为这个项目的核心逻辑补充测试:先了解现有测试方式(框架/命令),再挑最值得测的部分写测试,最后运行确认全部通过。",
      en: "Add tests for the core logic: first learn the existing test setup (framework/commands), write tests for the most valuable paths, then run them until everything passes.",
    },
  },
  {
    name: "fix",
    desc: { zh: "运行测试/构建,修复所有失败", en: "Run tests/build, fix all failures" },
    prompt: {
      zh: "运行这个项目的测试和构建,找出所有失败/报错,逐个定位根因并修复,直到测试与构建全部通过。",
      en: "Run this project's tests and build, find every failure, diagnose the root cause of each, and fix them until tests and build are fully green.",
    },
  },
  {
    name: "explain",
    desc: { zh: "讲解项目结构与关键流程(只读)", en: "Explain the architecture (read-only)" },
    prompt: {
      zh: "阅读这个项目并讲解:整体架构、目录结构、数据/控制流,以及 2-3 个最核心的实现点。只读分析,不要修改任何文件。",
      en: "Read this project and explain it: overall architecture, layout, data/control flow, and the 2-3 most important implementation details. Read-only — do not modify any files.",
    },
  },
  {
    name: "commit",
    desc: { zh: "检查改动并写出规范的 git 提交", en: "Stage & commit current changes" },
    prompt: {
      zh: "用 git status / git diff 查看当前改动,按逻辑分组,写出符合项目风格的提交信息并完成 git commit(不要 push)。",
      en: "Inspect current changes with git status / git diff, group them logically, write commit messages matching the repo's style, and git commit (do not push).",
    },
  },
];
