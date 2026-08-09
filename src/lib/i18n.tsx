import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";

/// UI locales. zh/en are the app's native pair and REQUIRED on every entry
/// (compiler-enforced); community locales are optional per-key and fall back
/// to English, so partial coverage never breaks a build or blanks a label.
/// The agent/model layer stays zh|en by design — map with `agentLang()` at
/// that boundary (prompt quality follows model training data, not the UI).
export type Lang = "zh" | "en" | "pt";

/// What the agent/model layer speaks. Community UI locales ride the English
/// prompt path.
export function agentLang(lang: Lang): "zh" | "en" {
  return lang === "zh" ? "zh" : "en";
}

/// Selector metadata for the language switch — add a row here (and the
/// optional field in Entry) to introduce a locale.
export const LANGS: { id: Lang; label: string }[] = [
  { id: "zh", label: "中文" },
  { id: "en", label: "English" },
  { id: "pt", label: "Português (BR)" },
];

const LANG_KEY = "chaty.lang";

type Entry = { zh: string; en: string; pt?: string };

export const T = {
  // titlebar / sidebar
  newChat: { zh: "新对话", en: "New chat", pt: "Novo chat" },
  resizeSidebar: { zh: "拖动调节宽度，双击复位", en: "Drag to resize · double-click to reset", pt: "Arraste para redimensionar · clique duplo para restaurar" },
  // Code mode (agentic coding)
  modeChat: { zh: "对话", en: "Chat", pt: "Chat" },
  modeCode: { zh: "编程", en: "Code", pt: "Código" },
  cmNewSession: { zh: "新会话", en: "New session", pt: "Nova sessão" },
  cmNoSessions: { zh: "暂无会话", en: "No sessions yet", pt: "Nenhuma sessão ainda" },
  cmOpenFolder: { zh: "打开工作区…", en: "Open workspace…", pt: "Abrir workspace…" },
  cmBypass: { zh: "自动放行", en: "Bypass", pt: "Ignorar aprovação" },
  cmBypassHint: {
    zh: "开启后,写文件与执行命令不再逐条确认(谨慎)。",
    en: "When on, file writes and commands run without per-step approval (careful).", pt: "Quando ativado, gravações de arquivos e comandos são executados sem aprovação passo a passo (cuidado).",
  },
  cmWelcome: { zh: "编程智能体", en: "Coding agent", pt: "Agente de programação" },
  cmWelcomeReady: {
    zh: "描述你的编码任务,它会读代码、改文件、跑命令来完成。",
    en: "Describe a coding task — it reads code, edits files and runs commands to do it.", pt: "Descreva uma tarefa de programação — ele lê códigos, edita arquivos e executa comandos para concluí-la.",
  },
  cmWelcomePick: { zh: "先打开一个工作区文件夹开始。", en: "Open a workspace folder to start.", pt: "Abra uma pasta de workspace para começar." },
  cmPlaceholder: { zh: "交给它一个编码任务,Enter 发送…", en: "Give it a coding task — Enter to send…", pt: "Dê uma tarefa de programação a ele — Enter para enviar…" },
  cmPlaceholderMod: { zh: "交给它一个编码任务,⌘/Ctrl+Enter 发送…", en: "Give it a coding task — ⌘/Ctrl+Enter to send…", pt: "Dê uma tarefa de programação a ele — ⌘/Ctrl+Enter para enviar…" },
  cmPlaceholderNoWs: { zh: "先在上方打开工作区", en: "Open a workspace first (top)", pt: "Abra um workspace primeiro (topo)" },
  cmApproveBash: { zh: "允许执行命令?", en: "Run this command?", pt: "Executar este comando?" },
  cmApproveWrite: { zh: "允许改动文件?", en: "Apply this change?", pt: "Aplicar esta alteração?" },
  cmDeny: { zh: "拒绝", en: "Deny", pt: "Negar" },
  cmAllow: { zh: "允许", en: "Allow", pt: "Permitir" },
  cmAllowAlwaysCmd: { zh: "总是允许 “{cmd}”", en: "Always allow \"{cmd}\"", pt: "Sempre permitir \"{cmd}\"" },
  cmAllowAlwaysEdits: { zh: "总是允许文件修改", en: "Always allow file edits", pt: "Sempre permitir edições de arquivo" },
  cmAllowlist: { zh: "命令白名单", en: "Command allowlist", pt: "Allowlist de comandos" },
  cmAllowlistPh: { zh: "如 npm test、cargo build…", en: "e.g. npm test, cargo build…", pt: "ex: npm test, cargo build…" },
  cmAllowlistHint: {
    zh: "以这些前缀开头的命令无需逐条审批。审批弹窗里的「总是允许」只在当前会话内生效;这里的白名单永久生效。",
    en: "Commands starting with these prefixes run without approval. \"Always allow\" in the dialog is session-only; this list is permanent.", pt: "Comandos iniciados com esses prefixos são executados sem aprovação. \"Sempre permitir\" na caixa de diálogo vale apenas para a sessão atual; esta allowlist é permanente.",
  },
  cmRunning: { zh: "运行中", en: "Working", pt: "Processando" },
  cmThinking: { zh: "思考中…", en: "Thinking…", pt: "Pensando…" },
  cmThought: { zh: "已思考", en: "Reasoned", pt: "Raciocinou" },
  cmThinkOff: { zh: "不思考", en: "Off", pt: "Desligado" },
  cmThinkNormal: { zh: "标准", en: "Normal", pt: "Normal" },
  cmThinkDeep: { zh: "深入", en: "Deep", pt: "Profundo" },
  cmThinkHint: {
    zh: "思考强度:不思考最快;深入更周全但更慢。思考内容会显式展示。",
    en: "Reasoning depth: Off is fastest; Deep is more thorough but slower. Reasoning is shown.", pt: "Profundidade de raciocínio: Desligado é o mais rápido; Profundo é o mais minucioso, porém mais lento. O raciocínio é exibido.",
  },
  cmPlan: { zh: "任务计划", en: "Task plan", pt: "Plano de tarefa" },
  cmAskUser: { zh: "需要你来决定", en: "Your call", pt: "Você decide" },
  cmContext: { zh: "上下文用量", en: "Context used", pt: "Contexto usado" },
  cmAskCustom: { zh: "自定义回答…", en: "Custom answer…", pt: "Resposta personalizada…" },
  cmSlashClear: { zh: "新建会话(清空上下文)", en: "New session (clear context)", pt: "Nova sessão (limpar contexto)" },
  cmSlashThink: { zh: "思考强度", en: "Reasoning depth", pt: "Profundidade de raciocínio" },
  cmSlashBypass: { zh: "切换自动批准", en: "Toggle auto-approve", pt: "Alternar aprovação automática" },
  cmSlashHelp: { zh: "显示可用命令", en: "Show commands", pt: "Mostrar comandos" },
  cmLines: { zh: "行", en: "lines", pt: "linhas" },
  cmDiffMore: { zh: "… 共 {n} 处改动,已折叠余下部分", en: "… {n} changed lines total, rest collapsed", pt: "… {n} linhas alteradas no total, o resto foi recolhido" },
  cmEditN: { zh: "修改 {n}", en: "edit {n}", pt: "editar {n}" },
  cmMaxSteps: { zh: "单轮最大步数", en: "Max steps per turn", pt: "Máx. de passos por turno" },
  cmMaxStepsHint: {
    zh: "Code 模式一轮任务最多执行的工具步数。达到上限会暂停,回复「继续」可接着做。复杂任务可调高。",
    en: "How many tool steps a Code-mode turn may take before pausing. Reply \"continue\" to resume. Raise it for complex tasks.", pt: "Quantos passos um turno do modo Code pode realizar antes de pausar. Responda \"continue\" para retomar. Aumente para tarefas complexas.",
  },
  cmBashTimeout: { zh: "命令超时", en: "Command timeout", pt: "Tempo limite de comandos" },
  cmBashTimeoutHint: {
    zh: "bash 命令的默认超时时间。跑测试/构建较慢的项目可调高。",
    en: "Default timeout for bash commands. Raise it for slow test/build steps.", pt: "Tempo limite padrão para comandos bash. Aumente para etapas lentas de testes/build.",
  },
  cmPrefill: { zh: "处理提示词", en: "Processing prompt", pt: "Processando prompt" },
  cmDirAskTitle: { zh: "想访问工作区以外的目录", en: "Wants access outside the workspace", pt: "Requisita acesso fora do workspace" },
  cmDirAskHint: {
    zh: "允许后,该目录在本会话内可持续读写(顶部会显示,可随时一键取消)。拒绝则模型会换别的做法。",
    en: "If allowed, this directory stays accessible for the session (shown in the header, revocable anytime). If denied, the model will try another way.", pt: "Se permitido, este diretório permanecerá acessível durante a sessão (exibido no cabeçalho, revogável a qualquer momento). Se recusado, o modelo tentará outra abordagem.",
  },
  cmDirAllow: { zh: "允许访问", en: "Allow access", pt: "Permitir acesso" },
  cmSudoTitle: { zh: "高危:请求以 sudo 运行命令", en: "High risk: run a command with sudo", pt: "Alto risco: executar um comando com sudo" },
  cmSudoHint: {
    zh: "该命令以管理员权限执行,可能影响整个系统,超出工作区沙箱的保护范围。只有你确认安全时才允许。",
    en: "This runs with administrator privileges — it can affect your whole system and is outside the workspace sandbox. Allow only if you're sure it's safe.", pt: "Isso é executado com privilégios de administrador — pode afetar todo o seu sistema e está fora do sandbox do workspace. Permita apenas se tiver certeza de que é seguro.",
  },
  cmSudoAllow: { zh: "确认以 sudo 运行", en: "Run with sudo", pt: "Executar com sudo" },
  cmSudoPwPlaceholder: { zh: "sudo 密码(可留空)", en: "sudo password (optional)", pt: "senha do sudo (opcional)" },
  cmSudoPwNote: {
    zh: "密码仅本机使用:经加密通道直接喂给 sudo,不会显示、不写入日志或历史、也不发给模型。若系统已配置免密可留空。",
    en: "Used locally only: piped straight to sudo, never shown, logged, saved to history, or sent to the model. Leave blank if your system is passwordless.", pt: "Usado apenas localmente: enviado diretamente para o sudo, nunca exibido, registrado, salvo no histórico ou enviado ao modelo. Deixe em branco se o seu sistema não exige senha.",
  },
  cmGrantRevoke: { zh: "取消该目录的访问授权", en: "Revoke access to this directory", pt: "Revogar acesso a este diretório" },
  cmGrantAddTip: {
    zh: "允许本会话访问工作区以外的其他文件夹",
    en: "Allow this session to access another folder outside the workspace", pt: "Permitir que esta sessão acesse outra pasta fora do workspace",
  },
  cmTemp: { zh: "步骤温度", en: "Step temperature", pt: "Temperatura do passo" },
  cmTempHint: {
    zh: "Code 模式每步生成的采样温度。低 = 更稳定可复现(默认 0.3),高 = 更有创造性但更容易跑偏。",
    en: "Sampling temperature for each agent step. Lower = steadier and more reproducible (default 0.3); higher = more creative but less predictable.", pt: "Temperatura de amostragem para cada passo do agente. Mais baixa = mais constante e reprodutível (padrão 0.3); mais alta = mais criativa, porém menos previsível.",
  },
  cmThinkBudget: { zh: "思考预算 (tokens)", en: "Think budget (tokens)", pt: "Orçamento de raciocínio (tokens)" },
  cmThinkBudgetOff: { zh: "不限", en: "off", pt: "desligado" },
  cmThinkBudgetHint: {
    zh: "每步思考的 token 上限。超出后思考块被温和收束:已有思考保留在上下文里,模型基于它直接行动——不丢内容、不断连贯。0 = 不限制(单步生成上限自然封顶)。低温下容易无限思考的模型建议设 1000-2000。",
    en: "Hard per-step ceiling on thinking tokens. Over budget the think block closes gracefully: the reasoning stays in context and the model acts on it — nothing discarded, coherence kept. 0 = no ceiling (the per-step generation limit still bounds it). For models that loop in thought at low temperature, 1000-2000 works well.", pt: "Limite rígido de tokens de raciocínio por passo. Ao exceder o orçamento, o bloco de raciocínio é fechado de forma suave: o raciocínio permanece no contexto e o modelo age com base nele — nada é descartado, a coerência é mantida. 0 = sem teto (o limite de geração por passo ainda restringe). Para modelos que entram em loop de raciocínio em baixas temperaturas, 1000-2000 costuma funcionar bem.",
  },
  cmMaxTokens: { zh: "单步生成上限 (tokens)", en: "Per-step output limit (tokens)", pt: "Limite de saída por passo (tokens)" },
  cmMaxTokensAuto: { zh: "自动", en: "auto", pt: "automático" },
  cmMaxTokensHint: {
    zh: "每个 agent 步骤的生成 token 上限。0 = 按思考深度自动(关 4096 / 标准 6144 / 深度 8192);任何值都会被上下文窗口自动收紧。调低可以防长跑,调高给长思考和大文件写入留空间。",
    en: "Generation cap per agent step. 0 = auto by think depth (off 4096 / normal 6144 / deep 8192); any value is clamped to the context window. Lower it to bound runaways, raise it for long reasoning and big file writes.", pt: "Limite de geração por passo do agente. 0 = automático pela profundidade de raciocínio (desligado 4096 / normal 6144 / profundo 8192); qualquer valor é limitado pela janela de contexto. Diminua para restringir fugas, aumente para raciocínios longos e escritas de arquivos grandes.",
  },
  cmAutoEdits: { zh: "自动批准文件编辑", en: "Auto-approve file edits", pt: "Auto-aprovar edições de arquivos" },
  cmAutoEditsHint: {
    zh: "写入/编辑文件不再逐次询问(命令仍需批准)。每轮开始有检查点,可随时回滚。",
    en: "Write/edit steps run without asking (commands still need approval). Checkpoints let you roll back anytime.", pt: "Passos de escrita/edição rodam sem perguntar (comandos ainda precisam de aprovação). Checkpoints permitem que você reverta a qualquer momento.",
  },
  copyMsg: { zh: "复制这条消息", en: "Copy this message", pt: "Copiar esta mensagem" },
  jumpLatest: { zh: "回到最新", en: "Jump to latest", pt: "Pular para o mais recente" },
  expandAll: { zh: "展开全部", en: "Show all", pt: "Mostrar tudo" },
  collapseText: { zh: "收起", en: "Collapse", pt: "Recolher" },
  codeLines: { zh: "行", en: "lines", pt: "linhas" },
  chatCollapseCode: { zh: "长代码块自动折叠", en: "Auto-collapse long code blocks", pt: "Auto-recolher blocos de código longos" },
  chatCollapseCodeHint: {
    zh: "长代码块收起为标题+前几行预览,点击展开。",
    en: "Long code blocks fold to a header + preview; click to expand.", pt: "Blocos de código longos são recolhidos para um cabeçalho + prévia; clique para expandir.",
  },
  cmAutoReadOnly: { zh: "只读命令自动放行", en: "Auto-run read-only commands", pt: "Auto-executar comandos read-only" },
  cmAutoReadOnlyHint: {
    zh: "ls、grep、git log 这类明确只读的命令不再弹批准框;写入、删除或拿不准的命令仍会询问。",
    en: "Obviously read-only commands (ls, grep, git log) run without asking; anything that writes, deletes, or is uncertain still asks.", pt: "Comandos obviamente de leitura (ls, grep, git log) rodam sem perguntar; qualquer coisa que grave, apague ou seja incerta, continuará perguntando.",
  },
  cmHeadless: { zh: "后台运行浏览器", en: "Run browser hidden", pt: "Executar navegador em modo Headless" },
  cmHeadlessHint: {
    zh: "浏览器工具在后台运行,不弹出窗口。下次启动浏览器时生效;登录状态仍会保留。",
    en: "Browser tools run headless with no visible window. Applies the next time the browser starts; logins are kept.", pt: "As ferramentas do navegador rodam em segundo plano (headless) sem janela visível. Aplicável na próxima vez que o navegador for iniciado; logins são mantidos.",
  },
  cmSkills: { zh: "自定义技能", en: "Custom skills", pt: "Skills personalizadas" },
  cmBuiltinSkills: { zh: "内置技能", en: "Built-in skills", pt: "Skills integradas" },
  cmContinue: { zh: "继续", en: "Continue", pt: "Continuar" },
  cmRewind: { zh: "回滚", en: "Rewind", pt: "Reverter" },
  cmRewindHint: {
    zh: "回到此消息之前(恢复被改动的文件,并移除之后的对话)",
    en: "Rewind to before this message (restores edited files, removes later messages)", pt: "Reverter para o momento anterior a esta mensagem (restaura arquivos editados e remove as mensagens posteriores)",
  },
  cmRewindConfirm: {
    zh: "回到此消息之前?此后所有由智能体写入/编辑的文件将恢复原样,之后的对话会被移除。(bash 命令的副作用无法恢复)",
    en: "Rewind to before this message? All files the agent wrote/edited after this point will be restored and later messages removed. (bash side effects can't be undone)", pt: "Reverter para o momento anterior a esta mensagem? Todos os arquivos que o agente escreveu/editou após este ponto serão restaurados e as mensagens posteriores removidas. (Efeitos colaterais do bash não podem ser desfeitos)",
  },
  cmQueuePlaceholder: { zh: "运行中 — 输入将排队,本轮结束后自动执行…", en: "Working — messages queue up and run after this turn…", pt: "Trabalhando — as mensagens entrarão na fila e serão executadas após este turno…" },
  cmBgJobs: { zh: "后台任务", en: "background", pt: "segundo plano" },
  cmBgKill: { zh: "全部终止", en: "Kill all", pt: "Encerrar tudo" },
  cmBgKillHint: { zh: "点击可终止全部后台任务", en: "Click to kill all background jobs", pt: "Clique para encerrar todos os jobs em segundo plano" },
  cmBgKillConfirm: {
    zh: "终止全部 {n} 个后台任务?(如智能体启动的 dev server)",
    en: "Kill all {n} background jobs (e.g. dev servers the agent started)?", pt: "Encerrar todos os {n} jobs em segundo plano (ex: servidores dev iniciados pelo agente)?",
  },
  cmEg1: { zh: "解释这个项目的结构和关键流程", en: "Explain this project's structure and key flows", pt: "Explique a estrutura deste projeto e os fluxos principais" },
  cmEg2: { zh: "运行测试,修复所有失败", en: "Run the tests and fix every failure", pt: "Execute os testes e corrija todas as falhas" },
  cmEg3: { zh: "审查代码,列出潜在 bug 清单", en: "Review the code and list likely bugs", pt: "Revise o código e liste prováveis bugs" },
  cmSkillNamePh: { zh: "技能名(将成为 /命令)", en: "Skill name (becomes a /command)", pt: "Nome da skill (torna-se um comando /)" },
  cmSkillPromptPh: { zh: "提示词模板:选中技能后会填入输入框,可再补充细节…", en: "Prompt template — inserted into the composer when invoked…", pt: "Modelo de prompt — inserido na caixa de texto quando convocado…" },
  cmSkillsHint: {
    zh: "自定义可复用的任务模板。在 Code 输入框键入 / 即可调用,如 /review、/写测试。",
    en: "Reusable task templates. Type / in the Code composer to invoke, e.g. /review.", pt: "Modelos de tarefas reutilizáveis. Digite / na caixa de texto do modo Código para invocar, ex: /revisar.",
  },
  cmMcp: { zh: "MCP 服务器", en: "MCP servers", pt: "Servidores MCP" },
  cmMcpHint: {
    zh: "连接 MCP 服务器,把外部工具带给编码智能体:填命令(stdio)或 https:// 地址(Streamable HTTP)。未勾选「信任」的服务器,每次工具调用都需要你批准;工具结果一律按不可信内容处理。",
    en: "Connect MCP servers to bring external tools to the coding agent: a command (stdio) or an https:// URL (streamable HTTP). Unless marked trusted, every call needs your approval; results are always treated as untrusted content.", pt: "Conecte servidores MCP para disponibilizar ferramentas externas ao agente: um comando (stdio) ou uma URL https:// (HTTP contínuo). Ao menos que marcada como confiável, cada execução exigirá sua aprovação; os resultados são sempre tratados como conteúdo não confiável.",
  },
  cmMcpNamePh: { zh: "别名(如 gh)", en: "alias (e.g. gh)", pt: "apelido (ex: gh)" },
  cmMcpCmdPh: { zh: "命令 或 https:// URL", en: "command or https:// URL", pt: "comando ou URL https://" },
  cmMcpTokenPh: { zh: "Bearer 令牌(可选,仅 HTTP)", en: "Bearer token (optional, HTTP only)", pt: "Token Bearer (opcional, apenas HTTP)" },
  cmMcpTrusted: { zh: "信任", en: "trusted", pt: "confiável" },
  cmMemory: { zh: "项目记忆", en: "Project memory", pt: "Memória do projeto" },
  cmMemoryHint: {
    zh: "让智能体把对后续会话有用的非显而易见发现存进 <工作区>/.chaty/memory/(纯 Markdown,人可编辑,永不上云);新会话开始时把索引带进上下文。关掉则不加载也不提供记忆工具。",
    en: "Let the agent save non-obvious findings future sessions need into <workspace>/.chaty/memory/ (plain markdown, human-editable, never leaves the machine); the index rides into context at the start of a new session. Off = no index loaded and no memory tool offered.", pt: "Permite que o agente guarde descobertas não triviais necessárias para sessões futuras em <workspace>/.chaty/memory/ (apenas markdown, editável por humanos, nunca sai da máquina); o índice é trazido para o contexto no início de uma nova sessão. Desligado = nenhum índice carregado e a ferramenta de memória não é oferecida.",
  },
  cmMcpStore: { zh: "精选服务器", en: "Curated servers", pt: "Servidores curados" },
  cmMcpStoreHint: {
    zh: "每个条目版本钉死、权限透明,并通过 Chaty 的实连认证(cargo store_cert)。添加后可在上方列表里启停。",
    en: "Every entry is version-pinned, permission-transparent, and live-certified against Chaty's client (cargo store_cert). Once added, manage it in the list above.", pt: "Cada entrada tem sua versão fixada, permissões transparentes e é certificada dinamicamente pelo client do Chaty (cargo store_cert). Uma vez adicionado, você pode gerenciar pela lista acima.",
  },
  cmMcpCertified: { zh: "已认证", en: "certified", pt: "certificado" },
  cmMcpAddBtn: { zh: "添加", en: "Add", pt: "Adicionar" },
  cmMcpAdded: { zh: "已添加", en: "added", pt: "adicionado" },
  cmSkillFiles: { zh: "技能文件", en: "Skill files", pt: "Arquivos de skill" },
  cmSkillFilesHint: {
    zh: "技能 = 一份写着步骤的 Markdown。系统提示只带「名字+何时用」一行,模型需要时才调 use_skill 载入正文——所以技能再多也不占上下文。放在 ~/.chaty/skills/ (全局)或 项目/.chaty/skills/ (项目,同名覆盖全局)。下面是随应用附带的官方技能,可单独关闭。",
    en: "A skill is a markdown file of steps. The prompt carries only one line per skill (name + when); the body loads via use_skill only when needed — so skills cost almost no context. Put them in ~/.chaty/skills/ (global) or <project>/.chaty/skills/ (project, shadows global). Below are the ones bundled with Chaty; each can be turned off.", pt: "Uma skill é um arquivo markdown com passos. O prompt recebe apenas uma linha por skill (nome + uso); o corpo é carregado usando use_skill somente se houver necessidade — logo, skills quase não consomem contexto. Adicione-as em ~/.chaty/skills/ (global) ou <projeto>/.chaty/skills/ (projeto, substitui a global). Abaixo estão as incluídas com o Chaty; cada uma pode ser desativada.",
  },
  cmCompacted: { zh: "上下文已自动压缩", en: "Context auto-compacted", pt: "Contexto auto-comprimido" },
  cmdkGoCode: { zh: "切换到 Code 模式", en: "Switch to Code mode", pt: "Mudar para o modo Code" },
  cmdkGoChat: { zh: "切换到 Chat 模式", en: "Switch to Chat mode", pt: "Mudar para o modo Chat" },
  setCatGeneral: { zh: "通用", en: "General", pt: "Geral" },
  setCatChat: { zh: "对话", en: "Chat", pt: "Chat" },
  setCatSampling: { zh: "采样", en: "Sampling", pt: "Amostragem" },
  setCatModel: { zh: "模型", en: "Model", pt: "Modelo" },
  setCatVoice: { zh: "语音", en: "Voice", pt: "Voz" },
  setCatData: { zh: "数据", en: "Data", pt: "Dados" },
  setCatAbout: { zh: "关于", en: "About", pt: "Sobre" },
  aboutTagline: {
    zh: "本地、私密的 GGUF / MLX 模型桌面聊天应用",
    en: "Local, private desktop chat for GGUF & MLX models", pt: "App desktop offline e privado para chat com modelos GGUF & MLX",
  },
  aboutCheckUpdate: { zh: "检查更新", en: "Check for updates", pt: "Procurar atualizações" },
  aboutUpdateNow: { zh: "立即更新", en: "Update now", pt: "Atualizar agora" },
  aboutNewVersion: { zh: "发现新版本", en: "New version available", pt: "Nova versão disponível" },
  aboutUpToDate: { zh: "已是最新版本", en: "You're up to date", pt: "Você está atualizado" },
  cmWelcomeNoModel: { zh: "先在顶部加载一个模型,再开始编码任务。", en: "Load a model from the titlebar to start coding.", pt: "Carregue um modelo pelo topo para começar a programar." },
  cmPlaceholderNoModel: { zh: "请先加载模型…", en: "Load a model first…", pt: "Carregue um modelo primeiro…" },
  noModel: { zh: "未加载模型", en: "No model", pt: "Sem modelo" },
  loadingModel: { zh: "加载中…", en: "Loading…", pt: "Carregando…" },
  changeModel: { zh: "更换模型", en: "Change model", pt: "Trocar modelo" },
  modelsHeader: { zh: "models 文件夹", en: "Models folder", pt: "Pasta de modelos" },
  loadFromFolder: { zh: "从文件夹载入…", en: "Load from folder…", pt: "Carregar da pasta…" },
  noModelsFound: { zh: "models 文件夹中暂无模型", en: "No models in the models folder", pt: "Nenhum modelo na pasta models" },
  refreshModels: { zh: "刷新列表", en: "Refresh", pt: "Atualizar" },
  // command palette (⌘K)
  cmdkPlaceholder: { zh: "搜索命令、模型或对话…", en: "Search commands, models or chats…", pt: "Pesquisar comandos, modelos ou chats…" },
  cmdkHint: { zh: "打开命令面板", en: "for commands", pt: "para comandos" },
  cmdkEmpty: { zh: "没有匹配项", en: "No matches", pt: "Nenhuma correspondência" },
  cmdkChatHint: { zh: "对话", en: "Chat", pt: "Chat" },
  cmdkLoadModel: { zh: "加载模型：{name}", en: "Load model: {name}", pt: "Carregar modelo: {name}" },
  cmdkKbOn: { zh: "开启知识库", en: "Turn on knowledge base", pt: "Ligar base de conhecimento" },
  cmdkKbOff: { zh: "关闭知识库", en: "Turn off knowledge base", pt: "Desligar base de conhecimento" },
  cmdkWebOn: { zh: "开启联网搜索", en: "Turn on web search", pt: "Ligar pesquisa na web" },
  cmdkWebOff: { zh: "关闭联网搜索", en: "Turn off web search", pt: "Desligar pesquisa na web" },
  cmdkLive: { zh: "进入语音模式", en: "Start voice mode", pt: "Iniciar modo de voz" },
  // canvas (design studio)
  canvas: { zh: "画布", en: "Canvas", pt: "Canvas" },
  openInCanvas: { zh: "在画布中打开", en: "Open in Canvas", pt: "Abrir no Canvas" },
  canvasTitle: { zh: "设计画布", en: "Design canvas", pt: "Canvas de Design" },
  canvasCode: { zh: "代码", en: "Code", pt: "Código" },
  canvasReload: { zh: "刷新页面(重新运行脚本)", en: "Reload the page (re-runs scripts)", pt: "Recarregar página (reexecuta os scripts)" },
  canvasStop: { zh: "停止", en: "Stop", pt: "Parar" },
  canvasFollow: { zh: "↓ 跟随修改", en: "↓ Follow edits", pt: "↓ Seguir alterações" },
  canvasEditModeLabel: { zh: "画布 HTML 编辑模式", en: "Canvas HTML edit mode", pt: "Modo de edição HTML do Canvas" },
  canvasEditModeHint: {
    zh: "补丁:模型输出查找/替换块,快但要求逐字回显;整页重写:模型输出完整 HTML,系统随流式输出实时计算 diff——较小的模型建议用重写,更稳。",
    en: "Patch: the model emits search/replace blocks — fast, but needs verbatim echoes. Rewrite: the model streams the full HTML and the system diffs it live — the reliable choice for smaller models.", pt: "Patch: o modelo emite blocos de pesquisar/substituir — rápido, porém requer retorno idêntico. Reescrever: o modelo faz streaming do HTML completo e o sistema calcula a diferença (diff) ao vivo — a escolha mais confiável para modelos menores.",
  },
  canvasEditModePatch: { zh: "补丁", en: "Patch", pt: "Patch" },
  canvasEditModeRewrite: { zh: "整页重写", en: "Rewrite", pt: "Rewrite" },
  canvasDragHint: { zh: "拖动调宽 · 双击复位", en: "Drag to resize · double-click to reset", pt: "Arraste para redimensionar · clique duplo para restaurar" },
  canvasComposerEditing: { zh: "正在手动编辑——保存或取消后可继续对话", en: "Hand-editing — save or cancel to keep iterating", pt: "Edição manual — salve ou cancele para continuar iterando" },
  canvasSelPrefix: {
    zh: "请只修改以下选中的元素,其余保持不变:",
    en: "Modify ONLY these selected elements; keep everything else unchanged:", pt: "Modifique APENAS os elementos selecionados; mantenha todo o restante inalterado:",
  },
  canvasSelHint: { zh: "已选 {n} 个元素,修改将只落在选中处", en: "{n} element(s) selected — the edit targets them", pt: "{n} elemento(s) selecionado(s) — a edição será aplicada apenas a eles" },
  canvasSelClear: { zh: "清除选择", en: "Clear selection", pt: "Limpar seleção" },
  canvasEditCode: { zh: "编辑", en: "Edit", pt: "Editar" },
  canvasEditCodeHint: { zh: "手动编辑当前版本的代码,保存为新版本", en: "Hand-edit this version's code; saves as a new version", pt: "Edite este código manualmente; será salvo como uma nova versão" },
  canvasSaveEdit: { zh: "保存为新版本", en: "Save as new version", pt: "Salvar como nova versão" },
  canvasManualNote: { zh: "手动编辑", en: "Manual edit", pt: "Edição manual" },
  canvasConsole: { zh: "控制台", en: "Console", pt: "Console" },
  canvasConsoleEmpty: { zh: "当前版本没有控制台输出", en: "No console output for this version", pt: "Nenhuma saída no console para esta versão" },
  canvasReset: { zh: "重置画布", en: "Reset canvas", pt: "Resetar canvas" },
  canvasResetHint: { zh: "丢弃所有迭代版本,回到初始版本", en: "Drop every iteration and return to the first version", pt: "Descartar todas as iterações e retornar à primeira versão" },
  canvasResetConfirm: {
    zh: "将丢弃此画布的全部迭代版本,只保留初始版本。此操作不可撤销。",
    en: "All iterations of this canvas will be discarded, keeping only the first version. This cannot be undone.", pt: "Todas as iterações deste canvas serão descartadas, mantendo apenas a primeira versão. Esta ação não pode ser desfeita.",
  },
  canvasFull: { zh: "全屏", en: "Full screen", pt: "Tela cheia" },
  canvasExitFull: { zh: "退出全屏", en: "Exit full screen", pt: "Sair da tela cheia" },
  canvasScanning: { zh: "正在逐行修改…", en: "Rewriting line by line…", pt: "Reescrevendo linha por linha…" },
  canvasScanWaiting: { zh: "模型思考中,即将开始修改…", en: "Model is thinking — edits start shortly…", pt: "O modelo está pensando — as edições começarão em breve…" },
  canvasDiff: { zh: "变更", en: "Changes", pt: "Alterações" },
  canvasNoDiff: { zh: "首个版本没有可对比的变更", en: "The first version has nothing to compare", pt: "A primeira versão não tem alterações para comparar" },
  canvasInspect: { zh: "对照", en: "Inspect", pt: "Inspecionar" },
  canvasInspectHint: {
    zh: "开启后:在预览中悬停/点击元素,右侧代码定位到对应行;点代码行,预览中高亮对应元素。",
    en: "When on: hover/click an element in the preview to jump to its code line; click a code line to flash its element.", pt: "Quando ativado: passe o mouse/clique em um elemento na prévia para pular para a respectiva linha do código; clique numa linha de código para destacar seu elemento.",
  },
  canvasIterate: { zh: "描述要修改的内容，回车发送…", en: "Describe a change, press Enter…", pt: "Descreva uma alteração, pressione Enter…" },
  canvasSend: { zh: "发送", en: "Send", pt: "Enviar" },
  canvasExport: { zh: "导出 HTML", en: "Export HTML", pt: "Exportar HTML" },
  canvasOpenExt: { zh: "在浏览器打开", en: "Open in browser", pt: "Abrir no navegador" },
  canvasInitial: { zh: "初始版本", en: "Initial", pt: "Inicial" },
  canvasEdit: { zh: "修改", en: "Edit", pt: "Editar" },
  canvasFix: { zh: "修复", en: "Fix", pt: "Consertar" },
  canvasHealMsg: { zh: "页面运行出错，让模型修复？", en: "The page threw an error — let the model fix it?", pt: "A página emitiu um erro — quer que o modelo tente consertá-lo?" },
  canvasFixBtn: { zh: "修复", en: "Fix it", pt: "Conserte" },
  canvasIgnore: { zh: "忽略", en: "Ignore", pt: "Ignorar" },
  canvasMute: { zh: "本次不再提示", en: "Don't ask again", pt: "Não perguntar novamente" },
  canvasGenerating: { zh: "生成中…", en: "Generating…", pt: "Gerando…" },
  canvasNeedsModel: { zh: "请先加载模型", en: "Load a model first", pt: "Carregue um modelo primeiro" },
  canvasNoHtml: { zh: "未能从输出中解析出 HTML", en: "Couldn't parse HTML from the output", pt: "Não foi possível extrair o HTML da resposta do modelo" },
  ejectModel: { zh: "卸载模型（回到空状态）", en: "Eject model (back to empty)", pt: "Descarregar modelo (voltar ao início)" },
  deleteModelFile: { zh: "删除该模型文件", en: "Delete this model file" },
  confirmDeleteModel: {
    zh: "永久删除模型文件「{name}」？此操作不可撤销。",
    en: 'Permanently delete the model file "{name}"? This cannot be undone.',
  },
  // html preview
  closePreview: { zh: "关闭预览", en: "Close preview" },
  noConversations: { zh: "暂无历史会话", en: "No conversations yet" },
  deleteConv: { zh: "删除会话", en: "Delete conversation" },
  confirm: { zh: "确认", en: "Confirm" },
  confirmDelete: { zh: "删除", en: "Delete" },
  confirmDeleteConv: {
    zh: "确定要删除这个对话吗？",
    en: "Delete this conversation?",
  },
  confirmDeleteSession: {
    zh: "确定要删除这个编程会话吗？",
    en: "Delete this coding session?",
  },
  pinConv: { zh: "置顶", en: "Pin" },
  unpinConv: { zh: "取消置顶", en: "Unpin" },
  renameConv: { zh: "重命名", en: "Rename" },
  // empty state
  readyMsg: { zh: "我已就绪——完全本地运行，对话不出本机。", en: "Ready — everything runs locally, nothing leaves your machine." },
  loadToStart: { zh: "加载一个本地模型，即可开始对话。", en: "Load a local model to start chatting." },
  // composer
  inputPh: { zh: "输入消息，Enter 发送，Shift+Enter 换行", en: "Message… (Enter to send, Shift+Enter for newline)" },
  inputPhMod: { zh: "输入消息，⌘/Ctrl+Enter 发送，Enter 换行", en: "Message… (⌘/Ctrl+Enter to send, Enter for newline)" },
  inputPhWeb: { zh: "联网搜索已开启，输入问题…", en: "Web search on — ask anything…" },
  inputPhNoModel: { zh: "先加载一个模型…", en: "Load a model first…" },
  webDesignOff: { zh: "网页设计模式：已关闭（/webdesign 切换）", en: "Web design mode: off (/webdesign to toggle)" },
  webDesignChip: { zh: "网页设计模式", en: "Web design mode" },
  inputPhDesign: { zh: "描述你想要的界面，模型会生成单文件 HTML…", en: "Describe the UI you want — get a single-file HTML…" },
  toolsMenu: { zh: "工具", en: "Tools" },
  toolAttach: { zh: "添加附件", en: "Attach a file" },
  toolWeb: { zh: "联网搜索", en: "Web search" },
  toolThink: { zh: "思考模式", en: "Thinking mode" },
  toolDesign: { zh: "网页设计模式", en: "Web design mode" },
  thinkUnsupported: { zh: "当前模型不支持思考模式", en: "This model doesn't support thinking" },
  // update banner
  updateAvailable: { zh: "发现新版本 v{v}", en: "Update available — v{v}" },
  updateNow: { zh: "立即更新", en: "Update now" },
  updateLater: { zh: "稍后", en: "Later" },
  updateDownloading: { zh: "下载中…", en: "Downloading…" },
  // context usage
  ctxUsage: { zh: "上下文占用", en: "Context usage" },
  stopTitle: { zh: "停止生成", en: "Stop" },
  sendTitle: { zh: "发送", en: "Send" },
  micStart: { zh: "语音输入", en: "Voice input" },
  micStop: { zh: "停止录音", en: "Stop recording" },
  liveStart: { zh: "实时语音对话", en: "Live voice chat" },
  liveExit: { zh: "退出实时模式", en: "Exit live mode" },
  liveListening: { zh: "聆听中…", en: "Listening…" },
  liveThinking: { zh: "思考中…", en: "Thinking…" },
  liveSpeaking: { zh: "回答中…", en: "Speaking…" },
  speakAloud: { zh: "朗读回复", en: "Read replies aloud" },
  // context menu
  ctxCut: { zh: "剪切", en: "Cut" },
  ctxCopy: { zh: "复制", en: "Copy" },
  ctxPaste: { zh: "粘贴", en: "Paste" },
  ctxSelectAll: { zh: "全选", en: "Select all" },
  // window controls
  minimize: { zh: "最小化", en: "Minimize" },
  maximize: { zh: "最大化", en: "Maximize" },
  restore: { zh: "向下还原", en: "Restore" },
  close: { zh: "关闭", en: "Close" },
  // message actions
  sources: { zh: "来源", en: "Sources" },
  copy: { zh: "复制", en: "Copy" },
  fork: { zh: "分叉", en: "Branch" },
  copyTitle: { zh: "复制回答", en: "Copy reply" },
  forkTitle: { zh: "从这里分叉为新对话", en: "Branch into a new chat from here" },
  reread: { zh: "朗读", en: "Replay" },
  rereadStop: { zh: "停止", en: "Stop" },
  rereadTitle: { zh: "朗读这条回复", en: "Read this reply aloud" },
  regenerate: { zh: "重新生成", en: "Regenerate" },
  regenTitle: { zh: "重新生成这条回答（会丢弃其后的对话）", en: "Regenerate this reply (drops what follows)" },
  editMsg: { zh: "编辑", en: "Edit" },
  saveEdit: { zh: "保存并重发", en: "Save & resend" },
  cancel: { zh: "取消", en: "Cancel" },
  // assistant message
  searching: { zh: "正在联网搜索…", en: "Searching the web…" },
  searchingKb: { zh: "正在检索知识库…", en: "Searching the knowledge base…" },
  searchingMix: { zh: "正在检索知识库与网络…", en: "Searching knowledge base & web…" },
  composing: { zh: "正在整理上文…", en: "Composing earlier context…" },
  contextSummary: {
    zh: "【早前对话摘要，供延续参考】\n",
    en: "[Summary of earlier conversation, for continuity]\n",
  },
  thinking: { zh: "正在思考", en: "Thinking" },
  thoughtExpand: { zh: "已深度思考 · 点击展开", en: "Reasoned · click to expand" },
  thoughtCollapse: { zh: "已深度思考 · 点击收起", en: "Reasoned · click to collapse" },
  // attachment
  removeAttach: { zh: "移除附件", en: "Remove attachment" },
  visionAttach: { zh: "图片 · 模型将直接查看", en: "Image · the model will see it" },
  cmAttachFile: { zh: "附加文件(文档或图片)", en: "Attach a file (document or image)" },
  cmClickPreview: { zh: "点击预览", en: "click to preview" },
  cmSaveImage: { zh: "保存到本地", en: "Save to disk" },
  cmSaved: { zh: "已保存", en: "Saved" },
  cmScreenshot: { zh: "网页截图", en: "Screenshot" },
  attachContextLabel: { zh: "附件", en: "Attachment" },
  visionBadge: { zh: "视觉", en: "Vision" },
  visionBadgeTip: { zh: "支持视觉——加载后可直接理解图片", en: "Vision-capable — understands images once loaded" },
  mlxBadgeTip: { zh: "MLX 文件夹模型，由 Apple Silicon 专用引擎运行", en: "MLX folder model — runs on the Apple-Silicon-native engine" },
  storeSearchPh: { zh: "搜索模型，或粘贴仓库链接后回车…", en: "Search models, or paste a repo link and press Enter…" },
  storeFormat: { zh: "格式", en: "Format" },
  storeSort: { zh: "排序", en: "Sort" },
  storeTrending: { zh: "热门", en: "Trending" },
  storeDownloads: { zh: "下载量", en: "Downloads" },
  storeLikes: { zh: "点赞", en: "Likes" },
  storeUpdated: { zh: "最近更新", en: "Recently updated" },
  storeNoResults: { zh: "没有找到模型", en: "No models found" },
  storePickModel: { zh: "从左侧选择一个模型查看详情", en: "Pick a model on the left to see details" },
  storeFitsRam: { zh: "✓ 可完整载入内存", en: "✓ Fits fully in memory" },
  storeOverRam: { zh: "≈ 接近或超出内存，速度可能受影响", en: "≈ Near or over memory — may run slowly" },
  storeVisionIncluded: { zh: "自动附带视觉编码器", en: "vision encoder included" },
  storeReadmeEmpty: { zh: "该仓库没有 README", en: "This repo has no README" },
  storeToday: { zh: "今天", en: "today" },
  storeDaysAgo: { zh: "{n} 天前", en: "{n} days ago" },
  storeMlxMacOnly: { zh: "MLX 模型仅支持 macOS (Apple Silicon)——请选择 GGUF 版本", en: "MLX models are macOS (Apple Silicon) only — pick a GGUF build instead" },
  mmprojFailed: {
    zh: "视觉编码器加载失败，本次会话仅支持文本（图片将走 OCR）。",
    en: "The vision encoder failed to load — text only this session (images fall back to OCR).",
  },
  miVision: { zh: "视觉", en: "Vision" },
  setupVision: { zh: "支持看图", en: "Understands images" },
  truncatedSuffix: { zh: " · 已截断", en: " · truncated" },
  charsLabel: { zh: "{n} 字", en: "{n} chars" },
  readAttachFailed: { zh: "读取附件失败", en: "Failed to read file" },
  dropToAttach: { zh: "松开以添加为附件", en: "Drop to attach" },
  searchConv: { zh: "搜索对话…", en: "Search chats…" },
  noMatches: { zh: "无匹配对话", en: "No matching chats" },
  exportTitle: { zh: "导出对话", en: "Export chat" },
  exportMd: { zh: "导出为 Markdown", en: "Export as Markdown" },
  exportJson: { zh: "导出为 JSON", en: "Export as JSON" },
  exportFailed: { zh: "导出失败", en: "Export failed" },
  dlTitle: { zh: "下载模型", en: "Download model" },
  dlHint: {
    zh: "输入 HuggingFace 仓库（owner/name）或其网址，列出其中的模型文件（GGUF / MLX）后下载到模型文件夹。",
    en: "Enter a HuggingFace repo (owner/name) or URL to list its model files (GGUF / MLX) and download into your models folder.",
  },
  dlGet: { zh: "下载", en: "Download" },
  dlSearchFailed: { zh: "查找失败", en: "Lookup failed" },
  dlFailed: { zh: "下载失败", en: "Download failed" },
  // settings
  settingsTitle: { zh: "设置", en: "Settings" },
  language: { zh: "语言", en: "Language" },
  systemPrompt: { zh: "系统提示词", en: "System prompt" },
  systemPromptPh: { zh: "可选，例如：你是一个简洁、专业的助手。", en: "Optional, e.g. You are a concise, professional assistant." },
  temperature: { zh: "温度", en: "Temperature" },
  maxTokens: { zh: "最大生成长度", en: "Max length" },
  off: { zh: "关", en: "off" },
  repeatPenalty: { zh: "重复惩罚", en: "Repeat penalty" },
  stopSeqs: { zh: "停止词（每行一个）", en: "Stop sequences (one per line)" },
  stopSeqsPh: { zh: "例如：\nUser:\n###", en: "e.g.\nUser:\n###" },
  presets: { zh: "提示词预设", en: "Prompt presets" },
  presetNamePh: { zh: "预设名称…", en: "Preset name…" },
  presetSave: { zh: "保存当前", en: "Save" },
  voice: { zh: "语音发音人", en: "Voice" },
  voiceSpeed: { zh: "语速", en: "Speech rate" },
  theme: { zh: "主题", en: "Theme" },
  themeSystem: { zh: "跟随系统", en: "System" },
  themeLight: { zh: "浅色", en: "Light" },
  themeDark: { zh: "深色", en: "Dark" },
  resetDefaults: { zh: "恢复默认", en: "Reset" },
  // GPU acceleration settings
  gpuAccel: { zh: "GPU 加速", en: "GPU acceleration" },
  gpuAuto: { zh: "自动", en: "Auto" },
  gpuOff: { zh: "关闭", en: "Off" },
  gpuCustom: { zh: "自定义", en: "Custom" },
  gpuLayersLabel: { zh: "GPU 层数", en: "GPU layers" },
  gpuHint: {
    zh: "自动模式会按显存把尽量多的层放到 GPU。更改将在下次加载模型时生效。",
    en: "Auto fills the GPU with as many layers as VRAM allows. Changes apply on the next model load.",
  },
  // context length settings
  ctxLength: { zh: "上下文长度", en: "Context length" },
  ctxAuto: { zh: "自动", en: "Auto" },
  ctxTokens: { zh: "上下文 Token", en: "Context tokens" },
  ctxHint: {
    zh: "自动 = 在内存允许范围内尽量用满模型的训练上下文。自定义值也会按内存上限自动回落。",
    en: "Auto = as much of the model's trained context as memory allows. Custom values are also capped to fit memory.",
  },
  reloadApply: { zh: "重新加载模型以生效", en: "Reload model to apply" },
  // data management
  openDataDir: { zh: "打开数据文件夹", en: "Open data folder" },
  clearAllChats: { zh: "清空所有对话", en: "Clear all chats" },
  confirmClearChats: {
    zh: "确定要删除全部对话记录吗？此操作不可撤销。",
    en: "Delete all conversations? This cannot be undone.",
  },
  dataHint: {
    zh: "所有对话、模型与索引都保存在本地数据文件夹中，可手动复制备份。",
    en: "All conversations, models and indexes live in the local data folder — copy it to back up.",
  },
  dataFolder: { zh: "数据文件夹", en: "Data folder" },
  clearChatsHint: {
    zh: "删除全部对话与 Code 会话记录，不可撤销。",
    en: "Deletes every conversation and Code session. Irreversible.",
  },
  clearKb: { zh: "清空知识库", en: "Clear knowledge base" },
  clearKbHint: {
    zh: "移除全部已索引文档与向量数据，原始文件不受影响。",
    en: "Removes all indexed documents and vectors; original files are untouched.",
  },
  clearKbConfirm: {
    zh: "确定要清空知识库吗？所有已索引的文档都将被移除。",
    en: "Clear the knowledge base? All indexed documents will be removed.",
  },
  statConvs: { zh: "对话", en: "Conversations" },
  statMsgs: { zh: "消息", en: "Messages" },
  statCodeSessions: { zh: "Code 会话", en: "Code sessions" },
  statModels: { zh: "本地模型", en: "Local models" },
  statKbDocs: { zh: "知识库文档", en: "KB documents" },
  statKbChunks: { zh: "知识块", en: "KB chunks" },
  statDbSize: { zh: "数据库占用", en: "Database size" },
  // general / chat / model preferences
  setDarkScheme: { zh: "深色方案", en: "Dark palette" },
  setDarkSchemeHint: { zh: "深色模式使用的配色", en: "Colours used in dark mode" },
  setLightScheme: { zh: "浅色方案", en: "Light palette" },
  setLightSchemeHint: { zh: "浅色模式使用的配色", en: "Colours used in light mode" },
  schemeWarmCharcoal: { zh: "暖炭", en: "Warm charcoal" },
  schemeCoolCharcoal: { zh: "冷黑", en: "Cool charcoal" },
  schemePaper: { zh: "纸白", en: "Paper" },
  schemeCream: { zh: "暖米", en: "Cream" },
  setCodeTheme: { zh: "代码高亮", en: "Code highlighting" },
  setCodeThemeHint: { zh: "回答中代码块的配色", en: "Palette for code blocks in answers" },
  setUiScale: { zh: "界面缩放", en: "UI scale" },
  setUiScaleHint: { zh: "整体放大或缩小界面", en: "Zoom the whole interface" },
  setSendKey: { zh: "发送快捷键", en: "Send shortcut" },
  setSendKeyHint: {
    zh: "选择组合键时，Enter 换行",
    en: "With the combo, Enter inserts a newline",
  },
  setReduceMotion: { zh: "减少动效", en: "Reduce motion" },
  setReduceMotionHint: {
    zh: "关闭界面过渡与弹出动画",
    en: "Disable UI transitions and pop-in animations",
  },
  setAnswerSize: { zh: "回答字号", en: "Answer text size" },
  setAnswerSizeHint: { zh: "模型回答的阅读字号", en: "Reading size for model answers" },
  sizeSm: { zh: "紧凑", en: "Compact" },
  sizeMd: { zh: "标准", en: "Standard" },
  sizeLg: { zh: "舒适", en: "Comfortable" },
  setAutoTitle: { zh: "自动命名对话", en: "Auto-name conversations" },
  setAutoTitleHint: {
    zh: "首轮回复后自动生成标题",
    en: "Generate a title after the first reply",
  },
  setAutoLoadLast: { zh: "启动时加载上次模型", en: "Load last model on launch" },
  setAutoLoadLastHint: {
    zh: "打开应用即恢复上次使用的模型",
    en: "Restore the previously used model at startup",
  },
  modelsFolder: { zh: "模型文件夹", en: "Models folder" },
  modelsFolderHint: {
    zh: "放入 GGUF 文件或 MLX 模型文件夹即可被识别",
    en: "Drop GGUF files or MLX model folders here to make them available",
  },
  modelsDirHintToast: {
    zh: "将模型文件放入此文件夹后，重新打开模型选择器即可看到",
    en: "Drop model files into this folder, then reopen the model picker to see them",
  },
  hfEndpoint: { zh: "HuggingFace 端点", en: "HuggingFace endpoint" },
  hfEndpointOfficial: { zh: "官方", en: "Official" },
  hfEndpointCustom: { zh: "自定义", en: "Custom" },
  hfEndpointHint: {
    zh: "模型搜索、下载和知识库模型都走此端点。大陆网络推荐 hf-mirror.com；镜像不支持 xet 回退，个别文件失败时请切回官方。",
    en: "Model search, downloads and the knowledge-base model all use this endpoint. hf-mirror.com is recommended in mainland China; mirrors lack the xet fallback — switch back to Official if a file fails.",
  },
  tipHfEndpoint: {
    zh: "huggingface.co 不可达时可切换到与其路径兼容的镜像站",
    en: "Switch to a path-compatible mirror when huggingface.co is unreachable",
  },
  voicePreview: { zh: "试听", en: "Preview" },
  voicePreviewHint: {
    zh: "用当前语音和语速播放示例",
    en: "Play a sample with the current voice and speed",
  },
  voicePreviewBtn: { zh: "播放示例", en: "Play sample" },
  voiceEngineHint: {
    zh: "语音由本地 Whisper（识别）与 Kokoro（朗读）驱动，仅支持英文，完全离线。用于消息朗读与 Live 语音对话。",
    en: "Voice runs locally — Whisper for speech recognition, Kokoro for read-aloud. English only, fully offline. Powers message read-aloud and Live voice chat.",
  },
  // knowledge base (local RAG)
  kbTitle: { zh: "本地知识库", en: "Local knowledge base" },
  kbDocs: { zh: "个文档", en: "docs" },
  kbChunks: { zh: "个片段", en: "chunks" },
  kbModelNote: {
    zh: "首次使用需下载多语嵌入模型 bge-m3（约 730 MB，一次性，之后完全离线）。",
    en: "First use downloads the multilingual bge-m3 embedding model (~730 MB, one-time; fully offline after).",
  },
  kbDownloadModel: { zh: "下载嵌入模型", en: "Download embedding model" },
  kbEmpty: { zh: "还没有文档 — 添加 PDF / 文本 / 图片开始", en: "No documents yet — add a PDF / text / image file" },
  kbAdd: { zh: "添加文档", en: "Add documents" },
  kbAddFolder: { zh: "导入文件夹", en: "Import folder" },
  kbFolderEmpty: {
    zh: "该文件夹（含子目录）里没有可导入的文件。",
    en: "No importable files were found in that folder (or its subfolders).",
  },
  kbFolderConfirm: {
    zh: "在该文件夹及子目录中找到 {n} 个文件，全部导入知识库？",
    en: "Found {n} files in that folder and its subfolders. Import all of them?",
  },
  kbFolderConfirmGo: { zh: "全部导入", en: "Import all" },
  kbScopeTip: { zh: "勾选 = 参与检索", en: "Checked = included in retrieval" },
  kbScopeHint: {
    zh: "勾选要检索的文档，未勾选的不参与回答。",
    en: "Check the documents to search; unchecked ones are excluded from answers.",
  },
  kbRemove: { zh: "移除", en: "Remove" },
  kbIndexing: { zh: "索引中", en: "Indexing" },
  // deep-dive podcast (NotebookLM-style)
  kbPodcast: { zh: "生成深度播客", en: "Generate deep-dive podcast" },
  kbReport: { zh: "生成报告", en: "Generate report" },
  kbClear: { zh: "清空知识库", en: "Clear all" },
  kbClearConfirm: {
    zh: "确定要清空整个知识库吗？所有已索引的文档都会被移除（嵌入模型保留）。此操作无法撤销。",
    en: "Clear the entire knowledge base? All indexed documents will be removed (the embedding model is kept). This can't be undone.",
  },
  kbReportTitle: { zh: "知识库报告", en: "Knowledge-base report" },
  kbReportEmpty: {
    zh: "正在根据你的知识库内容生成概览报告……",
    en: "Generating an overview report from your knowledge base…",
  },
  kbReportRunning: {
    zh: "正在根据知识库内容智能生成报告（按文件引用，全程不联网）",
    en: "Generating a report from your knowledge base (cited per file, fully offline)",
  },
  kbReportRegen: { zh: "重新生成", en: "Regenerate" },
  kbReportFiles: { zh: "个文件", en: "files" },
  kbReportPhasePlan: { zh: "正在梳理知识库…", en: "Reviewing the knowledge base…" },
  kbReportPhaseRead: { zh: "正在读取文档内容…", en: "Reading the documents…" },
  kbReportNeedModel: {
    zh: "请先加载一个对话模型，再生成知识库报告。",
    en: "Load a chat model first, then generate the knowledge-base report.",
  },
  podcastTitle: { zh: "深度播客", en: "Deep-dive podcast" },
  podcastSub: {
    zh: "双主持人 · 英文 · 基于本地知识库",
    en: "Two hosts · English · from your knowledge base",
  },
  podcastIntro: {
    zh: "根据你启用的知识库文档，由模型撰写一段英文双人对话脚本，再用 Kokoro 一男一女两种音色交替朗读。生成期间其他 LLM 功能会暂时锁定，可随时取消，完成后可播放并导出音频。（仅生成英文播客）",
    en: "From your enabled documents, the model writes a two-host English script, then Kokoro reads it aloud with alternating male/female voices. Other LLM features are locked while it runs; you can cancel anytime, then play and export the audio. (English podcast only.)",
  },
  podcastStart: { zh: "开始生成", en: "Generate" },
  podcastWriting: { zh: "正在撰写脚本…", en: "Writing the script…" },
  podcastVoicing: { zh: "正在合成语音…", en: "Synthesizing voices…" },
  podcastEta: { zh: "预计剩余", en: "Time left" },
  etaLeft: { zh: "预计剩余", en: "Time left" },
  podcastCancel: { zh: "取消生成", en: "Cancel" },
  podcastTurns: { zh: "段对话", en: "turns" },
  podcastPlay: { zh: "播放", en: "Play" },
  podcastStop: { zh: "停止", en: "Stop" },
  podcastExport: { zh: "导出音频", en: "Export audio" },
  podcastRegen: { zh: "重新生成", en: "Regenerate" },
  podcastRetry: { zh: "重试", en: "Try again" },
  podcastNeedModel: { zh: "请先加载一个聊天模型", en: "Load a chat model first" },
  podcastNoScript: {
    zh: "脚本生成失败，请重试或更换模型",
    en: "Could not produce a usable script — try again or switch models",
  },
  // Rendered only for zh UI users (the panel gates on lang) — the en text
  // exists to keep the "en is the universal fallback" invariant airtight.
  podcastFootZh: {
    zh: "提示：播客内容为英文，适合用于英语听力与学习。",
    en: "Note: podcast output is in English.",
  },
  toolKb: { zh: "知识库检索", en: "Knowledge base" },
  toolKbManage: { zh: "管理知识库…", en: "Manage knowledge base…" },
  kbNeedSetup: { zh: "请先在知识库面板下载嵌入模型并添加文档", en: "Download the embedding model and add documents first" },
  // tools-menu groups (hover submenus)
  toolKbGroup: { zh: "知识库", en: "Knowledge base" },
  toolWebGroup: { zh: "联网功能", en: "Web" },
  // deep research
  toolDeepResearch: { zh: "深度研究", en: "Deep Research" },
  drTitle: { zh: "深度研究", en: "Deep Research" },
  drTopicPh: {
    zh: "输入研究主题，例如：固态电池的最新进展与商业化前景",
    en: "Enter a topic, e.g. the state of solid-state batteries and commercialization outlook",
  },
  drDepth: { zh: "深度", en: "Depth" },
  drDepthQuick: { zh: "快速（2 轮）", en: "Quick (2 rounds)" },
  drDepthStd: { zh: "标准（3 轮）", en: "Standard (3 rounds)" },
  drDepthDeep: { zh: "深入（4 轮）", en: "Deep (4 rounds)" },
  drRun: { zh: "开始研究", en: "Research" },
  drStop: { zh: "停止", en: "Stop" },
  drEmpty: {
    zh: "给出一个主题，模型会自动多轮检索网络、边查边推理，最后写成一篇带参考来源的深度报告，可导出 PDF。",
    en: "Give a topic; the model searches the web over several rounds, reasons as it goes, and writes a cited in-depth report you can export to PDF.",
  },
  drRound: { zh: "第", en: "round" },
  drQueries: { zh: "次检索", en: "searches" },
  drSources: { zh: "个来源", en: "sources" },
  drExportPdf: { zh: "导出 PDF", en: "Export PDF" },
  drExportMd: { zh: "导出 Markdown", en: "Export Markdown" },
  drBackToChat: { zh: "返回对话", en: "Back to chat" },
  drPhasePlanning: { zh: "正在规划检索方向…", en: "Planning the research…" },
  drPhaseSearching: { zh: "正在联网检索…", en: "Searching the web…" },
  drPhaseReasoning: { zh: "正在分析并寻找信息缺口…", en: "Analyzing and finding gaps…" },
  drPhaseWriting: { zh: "正在撰写报告…", en: "Writing the report…" },
  drPhaseDone: { zh: "完成", en: "Done" },
  // first-launch setup
  setupBtn: { zh: "一键配置", en: "Set up for me" },
  setupTitle: { zh: "为这台电脑挑选模型", en: "Models picked for this machine" },
  setupBudget: { zh: "可用模型内存约", en: "model memory budget ≈" },
  setupDownload: { zh: "下载", en: "Download" },
  setupResolving: { zh: "正在查找…", en: "Resolving…" },
  setupLoad: { zh: "加载此模型", en: "Load this model" },
  setupNotFound: { zh: "未找到合适的 GGUF 文件", en: "No suitable GGUF file found" },
  setupStoreLink: { zh: "想用社区的其他模型?去模型商店逛逛", en: "Want other community models? Browse the model store" },
  setupFoot: {
    zh: "推荐按本机内存自动匹配规模与量化，下载自 HuggingFace，存入应用的 models 文件夹。",
    en: "Sized & quantized for your memory, downloaded from HuggingFace into the app's models folder.",
  },
  // parameter tooltips
  tipTemperature: {
    zh: "随机性：越低回答越确定，越高越发散有创意（常用 0.7）",
    en: "Randomness: lower = more deterministic, higher = more creative (0.7 is typical)",
  },
  tipTopP: {
    zh: "核采样：只从累计概率前 P 的候选词中选择，与温度配合控制多样性",
    en: "Nucleus sampling: choose only from tokens within the top-P cumulative probability",
  },
  tipTopK: {
    zh: "只考虑概率最高的 K 个候选词；0 = 关闭此限制",
    en: "Consider only the K most likely tokens; 0 disables the limit",
  },
  tipMinP: {
    zh: "过滤相对概率低于最高候选 P 倍的词；0 = 关闭",
    en: "Drop tokens whose probability is below P× the top token's; 0 disables",
  },
  tipRepeatPenalty: {
    zh: "大于 1 时抑制重复用词；调得过高会伤害流畅度",
    en: ">1 discourages repetition; too high hurts fluency",
  },
  tipMaxTokens: {
    zh: "单次回复的最大 token 数；“不限制”时由上下文窗口决定",
    en: "Max tokens per reply; with no limit, the context window is the bound",
  },
  tipStopSeqs: {
    zh: "模型一旦输出这些字符串就立即停止生成",
    en: "Generation stops immediately when the model emits any of these strings",
  },
  tipGpuAccel: {
    zh: "放到 GPU 的网络层数；自动 = 按显存放尽量多的层",
    en: "How many layers run on the GPU; Auto fills as many as memory allows",
  },
  tipCtxLength: {
    zh: "模型能“记住”的对话长度（token）；越大越耗内存",
    en: "How much conversation the model can hold (tokens); larger uses more memory",
  },
  ejectingModel: { zh: "正在卸载旧模型…", en: "Ejecting old model…" },
  noLimit: { zh: "不限制", en: "No limit" },
  openModelsDir: { zh: "打开模型文件夹", en: "Open models folder" },
  // stop reasons (shown in the stats line after generation)
  stopEos: { zh: "自然结束", en: "finished" },
  stopLength: { zh: "达到长度上限", en: "length limit" },
  stopContext: { zh: "上下文已满", en: "context full" },
  stopStop: { zh: "命中停止词", en: "stop sequence" },
  stopCancelled: { zh: "已手动停止", en: "cancelled" },
  // hardware panel
  hwTitleBtn: { zh: "硬件信息", en: "Hardware" },
  hwTitle: { zh: "本机硬件", en: "Hardware" },
  hwCpu: { zh: "处理器", en: "CPU" },
  hwRam: { zh: "内存", en: "Memory" },
  hwGpu: { zh: "显卡", en: "GPU" },
  hwVram: { zh: "显存占用", en: "VRAM usage" },
  hwBackend: { zh: "GPU 后端", en: "GPU backend" },
  hwAccel: { zh: "当前模型加速", en: "Current model" },
  hwNoGpu: { zh: "未检测到独立显卡", en: "No discrete GPU detected" },
  hwThreads: { zh: "{n} 线程", en: "{n} threads" },
  hwLayersOn: { zh: "{a}/{b} 层在 GPU", en: "{a}/{b} layers on GPU" },
  hwCpuOnly: { zh: "纯 CPU 运行", en: "Running on CPU" },
  hwNoModel: { zh: "未加载模型", en: "No model loaded" },
  // model info panel
  miTitleBtn: { zh: "模型信息", en: "Model info" },
  miTitle: { zh: "模型信息", en: "Model info" },
  miName: { zh: "名称", en: "Name" },
  miArch: { zh: "架构", en: "Architecture" },
  miParams: { zh: "参数量", en: "Parameters" },
  miQuant: { zh: "量化", en: "Quantization" },
  miSize: { zh: "大小", en: "Size" },
  miContext: { zh: "上下文", en: "Context" },
  miTrained: { zh: "训练", en: "trained" },
  miLayers: { zh: "层数", en: "Layers" },
  miEmbed: { zh: "嵌入维度", en: "Embedding" },
  miEngine: { zh: "推理引擎", en: "Engine" },
  miTemplate: { zh: "对话模板", en: "Chat template" },
  miThinking: { zh: "思考推理", en: "Thinking" },
  miTools: { zh: "工具调用", en: "Tool calls" },
  miMultimodal: { zh: "多模态", en: "Multimodal" },
  miNoModel: { zh: "未加载模型", en: "No model loaded" },
  // load notices / OOM
  oomPartial: {
    zh: "显存不足，已自动减少 GPU 层数（{a}/{b} 层在 GPU）",
    en: "Low VRAM — GPU offload reduced to {a}/{b} layers",
  },
  oomCpu: { zh: "显存不足，已回退到 CPU 运行", en: "Low VRAM — fell back to CPU" },
  errorLog: { zh: "错误日志", en: "Error log" },
  errorLogHint: {
    zh: "应用异常会自动记录到 chaty-error.log。提 issue 时请附上这个文件，能大幅加快定位。",
    en: "App errors are recorded to chaty-error.log automatically. Please attach it when filing an issue — it speeds up diagnosis a lot.",
  },
  errorLogOpen: { zh: "打开错误日志", en: "Open error log" },
  conversionSuspect: {
    zh: "该模型家族与当前内置引擎的兼容性存在已知问题(转换元数据或引擎版本尚未跟上),回复可能退化为空白或乱码。macOS 上请改用该模型的 MLX 版本(运行正常);GGUF 需等待后续版本升级引擎。",
    en: "This model family has known compatibility issues with the bundled engine (conversion metadata or engine version lag) — replies may degenerate into blanks or noise. On macOS use the model's MLX build (works well); GGUF support awaits an engine upgrade.",
  },
  visionConfigMissing: {
    zh: "该模型目录缺少图像处理器配置(preprocessor_config.json),视觉功能已停用,本次以纯文本模式加载。从官方模型仓库补齐该文件后重新加载即可恢复视觉能力。",
    en: "This model folder is missing its image-processor configuration (preprocessor_config.json), so vision is disabled and the model was loaded text-only. Restore that file from the official model repo and reload to re-enable vision.",
  },
  gpuCrashCpu: {
    zh: "上次加载模型时 GPU 驱动导致程序崩溃，本次已改用 CPU 运行（速度较慢但稳定）。更新显卡驱动后可尝试恢复 GPU。",
    en: "The GPU driver crashed the app during the last model load — running on CPU this time (slower but stable). Update your GPU driver to try GPU again.",
  },
  ctxClamped: {
    zh: "上下文已按内存自动调整为 {n}（模型权重 + KV 缓存需放入统一内存）",
    en: "Context auto-fitted to {n} (weights + KV cache must fit in unified memory)",
  },
  oomFail: {
    zh: "内存不足，无法加载该模型。试试更小 / 更高量化的模型，或关闭其他占用内存的程序。",
    en: "Out of memory — couldn't load this model. Try a smaller / more-quantized model, or free up RAM.",
  },
  toastDismiss: { zh: "点击关闭", en: "Click to dismiss" },
  // greetings (time of day)
  greetMorning: { zh: "早上好", en: "Good morning" },
  greetNoon: { zh: "中午好", en: "Good afternoon" },
  greetAfternoon: { zh: "下午好", en: "Good afternoon" },
  greetEvening: { zh: "晚上好", en: "Good evening" },
  greetNight: { zh: "夜深了", en: "Working late" },
  // model-facing prompts
  todayNote: {
    zh: "当前日期是 {date}。涉及“今天/最近/现在”等时间时以此为准。",
    en: 'Today is {date}. When the question refers to "today/recent/now", use this date.',
  },
  webInstruction: {
    zh: "下面是联网检索到的资料，已按【1】【2】…编号。请综合它们，用自然连贯的语言直接回答用户的问题；在用到某条资料的句子末尾标注对应角标，如【1】或【1】【3】（不要写“来源”二字，只写数字角标）。若资料不足以回答，请直说。\n\n",
    en: "Below is information retrieved from the web, numbered 【1】【2】…. Use it to answer the user's question in natural prose, and append the matching citation marker(s) — e.g. 【1】 or 【1】【3】 — at the end of each sentence that draws on a source. If the material is insufficient, say so.\n\n",
  },
  ragInstruction: {
    zh: "下面是从用户本地知识库检索到的文档片段，已按【1】【2】…编号。严格依据这些片段回答：只陈述片段中明确支持的内容，绝不编造、不引入片段之外的事实或数字；若片段不足以回答，必须直接说明“当前文档未提及”。在用到某条片段的句子末尾标注对应角标，如【1】或【1】【3】（只写数字角标）。\n\n",
    en: "Below are passages retrieved from the user's local knowledge base, numbered 【1】【2】…. Answer STRICTLY from these passages: state only what they explicitly support, never invent facts or numbers beyond them; if they do not contain the answer, you must say the current documents do not mention it. Append the matching citation marker(s) — e.g. 【1】 or 【1】【3】 — at the end of each sentence that draws on a passage.\n\n",
  },
  attachInstruction: {
    zh: "用户上传了文件《{name}》，其内容如下，回答时请优先依据它：\n\n",
    en: 'The user uploaded a file "{name}". Its content is below; base your answer on it:\n\n',
  },
} satisfies Record<string, Entry>;

export type TKey = keyof typeof T;

function detectLang(): Lang {
  try {
    const saved = localStorage.getItem(LANG_KEY);
    if (LANGS.some((l) => l.id === saved)) return saved as Lang;
  } catch {
    /* ignore */
  }
  // Default first language is English (voice features are English-only and are
  // hidden when the user explicitly switches to Chinese).
  return "en";
}

/// Pure resolution (testable without the provider). Community locales are
/// allowed to be partial — a missing translation shows English, never a
/// blank or a raw key.
export function lookup(key: TKey, lang: Lang, vars?: Record<string, string | number>): string {
  const e: Entry | undefined = T[key];
  let s: string = e?.[lang] ?? e?.en ?? key;
  if (vars) {
    for (const k of Object.keys(vars)) s = s.replace(`{${k}}`, String(vars[k]));
  }
  return s;
}

interface I18n {
  lang: Lang;
  setLang: (l: Lang) => void;
  t: (key: TKey, vars?: Record<string, string | number>) => string;
}

const LangContext = createContext<I18n | null>(null);

export function LangProvider({ children }: { children: ReactNode }) {
  const [lang, setLang] = useState<Lang>(detectLang);

  useEffect(() => {
    try {
      localStorage.setItem(LANG_KEY, lang);
    } catch {
      /* ignore */
    }
  }, [lang]);

  const t = useCallback(
    (key: TKey, vars?: Record<string, string | number>) => lookup(key, lang, vars),
    [lang],
  );

  const value = useMemo<I18n>(() => ({ lang, setLang, t }), [lang, t]);
  return <LangContext.Provider value={value}>{children}</LangContext.Provider>;
}

export function useI18n(): I18n {
  const ctx = useContext(LangContext);
  if (!ctx) throw new Error("useI18n must be used within LangProvider");
  return ctx;
}
