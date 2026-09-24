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
  // Native reasoning-effort rungs. The model's own ladder, so the labels name
  // the rungs rather than Chaty's generic intensities — and a ladder that has
  // both `high` and `xhigh` needs two distinguishable names for them.
  effortLow: { zh: "低", en: "Low", pt: "Baixo" },
  effortMedium: { zh: "中", en: "Medium", pt: "Médio" },
  effortHigh: { zh: "高", en: "High", pt: "Alto" },
  effortXhigh: { zh: "最高", en: "Highest", pt: "Máximo" },
  effortHint: {
    zh: "该模型原生支持思考档位:低更快,高更周全(模型默认高)。",
    en: "This model has native reasoning-effort levels: low is faster, high is more thorough (the model's own default).",
    pt: "Este modelo tem níveis nativos de esforço de raciocínio: baixo é mais rápido, alto é mais completo (padrão do modelo).",
  },
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
    zh: "每步思考的 token 上限。超出后思考块被温和收束:已有思考保留在上下文里,模型基于它直接行动——不丢内容、不断连贯。0 = 不限制(单步生成上限自然封顶);滑杆上限即当前模型的上下文窗口。低温下容易无限思考的模型建议设 1000-2000。",
    en: "Hard per-step ceiling on thinking tokens. Over budget the think block closes gracefully: the reasoning stays in context and the model acts on it — nothing discarded, coherence kept. 0 = no ceiling (the per-step generation limit still bounds it); the slider tops out at the loaded model's context window. For models that loop in thought at low temperature, 1000-2000 works well.", pt: "Limite rígido de tokens de raciocínio por passo. Ao exceder o orçamento, o bloco de raciocínio é fechado de forma suave: o raciocínio permanece no contexto e o modelo age com base nele — nada é descartado, a coerência é mantida. 0 = sem teto (o limite de geração por passo ainda restringe); o controle vai até a janela de contexto do modelo carregado. Para modelos que entram em loop de raciocínio em baixas temperaturas, 1000-2000 costuma funcionar bem.",
  },
  cmMaxTokens: { zh: "单步生成上限 (tokens)", en: "Per-step output limit (tokens)", pt: "Limite de saída por passo (tokens)" },
  cmMaxTokensHint: {
    zh: "每个 agent 步骤的生成 token 上限。0 = 不限制(由上下文窗口自然封顶);滑杆上限即当前模型的上下文窗口。调低可以防长跑,调高给长思考和大文件写入留空间。",
    en: "Generation cap per agent step. 0 = no cap of its own (the context window still bounds it); the slider tops out at the loaded model's context window. Lower it to bound runaways, raise it for long reasoning and big file writes.", pt: "Limite de geração por passo do agente. 0 = sem limite próprio (a janela de contexto ainda restringe); o controle vai até a janela de contexto do modelo carregado. Diminua para restringir fugas, aumente para raciocínios longos e escritas de arquivos grandes.",
  },
  cmShell: { zh: "命令使用的终端", en: "Shell for commands", pt: "Shell para comandos" },
  cmShellHint: {
    zh: "智能体执行 bash 工具时使用的终端。Windows 默认是 cmd,而模型通常写 Linux/bash 命令——装了 Git Bash 或 PowerShell 就能在这里换。",
    en: "The shell the agent's bash tool runs commands in. Windows defaults to cmd while models write Linux/bash commands — pick Git Bash or PowerShell here if you have them.",
    pt: "O shell em que a ferramenta bash do agente executa comandos. No Windows o padrão é o cmd, mas os modelos escrevem comandos Linux/bash — escolha Git Bash ou PowerShell aqui, se os tiver.",
  },
  cmShellDefault: { zh: "默认", en: "Default", pt: "Padrão" },
  cmToolFormat: { zh: "工具调用格式", en: "Tool-call format", pt: "Formato das chamadas de ferramenta" },
  cmToolFormatHint: {
    zh: "模型写工具调用的格式。自动 = 用模型聊天模板里训练的那一种;手动选择则所有模型都用这一种。",
    en: "How the model writes tool calls. Auto = the format its chat template was trained on; a manual pick makes every model use that one.",
    pt: "Como o modelo escreve chamadas de ferramenta. Automático = o formato em que o template de chat dele foi treinado; uma escolha manual faz todos os modelos usarem esse.",
  },
  cmToolFormatAuto: { zh: "自动", en: "Auto", pt: "Automático" },
  cmToolFallback: { zh: "未知家族用", en: "Unknown families use", pt: "Famílias desconhecidas usam" },
  cmToolFallbackHint: {
    zh: "模型的聊天模板里没有写工具调用格式时用这一种。XML 的参数原样书写、不用转义,长代码不容易写坏。",
    en: "Used when a model's chat template names no tool-call format. XML takes every argument as written, nothing escaped, so long code is hard to break.",
    pt: "Usado quando o template de chat do modelo não define formato de chamada. O XML recebe cada argumento como escrito, sem escapes, então código longo dificilmente quebra.",
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
  cmBgKill: { zh: "全部终止", en: "Kill all", pt: "Encerrar tudo" },
  cmBgKillConfirm: {
    zh: "终止全部 {n} 个后台任务?(如智能体启动的 dev server)",
    en: "Kill all {n} background jobs (e.g. dev servers the agent started)?", pt: "Encerrar todos os {n} jobs em segundo plano (ex: servidores dev iniciados pelo agente)?",
  },
  cmBgCount: { zh: "{n} 个后台任务", en: "{n} background", pt: "{n} em segundo plano" },
  bgTitle: { zh: "后台任务", en: "Background tasks", pt: "Tarefas em segundo plano" },
  bgOpenHint: { zh: "查看后台任务的状态和输出", en: "See background tasks and their output", pt: "Ver as tarefas em segundo plano e a saída delas" },
  bgRunning: { zh: "运行中", en: "Running", pt: "Em execução" },
  bgFinished: { zh: "已结束", en: "Finished", pt: "Concluídas" },
  bgRunningNow: { zh: "运行中", en: "Running", pt: "Em execução" },
  bgDone: { zh: "已完成", en: "Completed", pt: "Concluída" },
  bgStopped: { zh: "已停止", en: "Stopped", pt: "Interrompida" },
  bgFailed: { zh: "失败 · 退出码 {code}", en: "Failed · exit {code}", pt: "Falhou · código {code}" },
  bgStop: { zh: "停止", en: "Stop", pt: "Parar" },
  bgClearFinished: { zh: "清除已结束", en: "Clear finished", pt: "Limpar concluídas" },
  bgNoOutput: { zh: "(暂无输出)", en: "(no output yet)", pt: "(ainda sem saída)" },
  bgEmpty: { zh: "没有后台任务", en: "No background tasks", pt: "Nenhuma tarefa em segundo plano" },
  bgClose: { zh: "关闭", en: "Close", pt: "Fechar" },
  cmChangesTitle: { zh: "已编辑 {n} 个文件", en: "Edited {n} files", pt: "{n} arquivos editados" },
  cmChangesUndoAll: { zh: "全部撤销", en: "Undo all", pt: "Desfazer tudo" },
  cmChangeUndo: { zh: "撤销", en: "Undo", pt: "Desfazer" },
  cmChangeUndone: { zh: "已撤销", en: "Undone", pt: "Desfeito" },
  cmChangeNew: { zh: "新建", en: "new", pt: "novo" },
  cmChangeDeleted: { zh: "已删除", en: "deleted", pt: "excluído" },
  cmChangeBinary: { zh: "二进制", en: "binary", pt: "binário" },
  cmChangeBinaryNote: { zh: "二进制文件,不显示差异。", en: "Binary file — no diff to show.", pt: "Arquivo binário — sem diferenças para mostrar." },
  cmChangeTooBig: { zh: "文件太大,没有保留差异内容。", en: "Too large — the diff was not kept.", pt: "Grande demais — as diferenças não foram guardadas." },
  cmUndoConfirmOne: {
    zh: "撤销后,{file} 会恢复成这一轮开始前的样子(这一轮新建的会被删除),这一轮对它的改动将无法找回。确定撤销吗?",
    en: "{file} will go back to how it was before this turn (or be removed, if the turn created it). This turn's changes to it can't be brought back. Undo?",
    pt: "{file} voltará a como estava antes deste turno (ou será removido, se o turno o criou). As alterações deste turno nele não poderão ser recuperadas. Desfazer?",
  },
  cmUndoConfirmMany: {
    zh: "撤销后,这 {n} 个文件会恢复成这一轮开始前的样子(这一轮新建的会被删除),这一轮对它们的改动将无法找回。确定撤销吗?",
    en: "These {n} files will go back to how they were before this turn (or be removed, if the turn created them). This turn's changes to them can't be brought back. Undo?",
    pt: "Estes {n} arquivos voltarão a como estavam antes deste turno (ou serão removidos, se o turno os criou). As alterações deste turno neles não poderão ser recuperadas. Desfazer?",
  },
  cmUndoFailed: { zh: "这些文件没能撤销:{files}", en: "Couldn't undo: {files}", pt: "Não foi possível desfazer: {files}" },
  cmLiveCardsOpen: {
    zh: "写入/编辑过程展开显示",
    en: "Show writes and edits as they happen",
    pt: "Mostrar gravações e edições enquanto acontecem",
  },
  cmLiveCardsOpenHint: {
    zh: "模型写入或编辑文件时,卡片展开并实时显示改动;关闭后卡片保持折叠,只显示标题和增删行数,点击可展开。",
    en: "While the model writes or edits a file, its card is open and the diff grows in view. Off, the card stays folded to its title and line counts; click to open it.",
    pt: "Enquanto o modelo grava ou edita um arquivo, o cartão fica aberto e o diff cresce à vista. Desligado, o cartão fica recolhido no título e nas contagens de linhas; clique para abrir.",
  },
  cmGroupByWs: { zh: "会话栏按工作区分组", en: "Group sessions by workspace", pt: "Agrupar sessões por espaço de trabalho" },
  cmGroupByWsHint: {
    zh: "左侧会话列表按各会话所在的工作区分组显示,分组可折叠。",
    en: "Groups the session list on the left by each session's workspace; groups can be folded.",
    pt: "Agrupa a lista de sessões à esquerda pelo espaço de trabalho de cada sessão; os grupos podem ser recolhidos.",
  },
  cmNoWorkspace: { zh: "未绑定工作区", en: "No workspace", pt: "Sem espaço de trabalho" },
  cmSessionWsMissing: {
    zh: "这个会话的工作区已经找不到了:{path}。请重新选择工作区。",
    en: "This session's workspace can no longer be found: {path}. Choose a workspace again.",
    pt: "O espaço de trabalho desta sessão não foi encontrado: {path}. Escolha um espaço de trabalho novamente.",
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
  cmSkillImport: { zh: "导入技能文件…", en: "Import skill files…", pt: "Importar arquivos de skill…" },
  cmSkillImportTitle: {
    zh: "选择技能文件(可多选:按住 Ctrl / ⌘ 点选)",
    en: "Pick skill files (several at once: hold Ctrl / ⌘)",
    pt: "Escolha arquivos de skill (vários: segure Ctrl / ⌘)",
  },
  cmSkillRemove: { zh: "删除这个技能文件", en: "Remove this skill file", pt: "Remover este arquivo de skill" },
  cmSkillFilesHint: {
    zh: "技能 = 一份写着步骤的 Markdown。系统提示只带「名字+何时用」一行,模型需要时才调 use_skill 载入正文——所以技能再多也不占上下文。放在 ~/.chaty/skills/ (全局)或 项目/.chaty/skills/ (项目,同名覆盖全局)。点「导入技能文件…」可以一次选一份或多份 .md(比如别的工具的 SKILL.md)复制进 ~/.chaty/skills/——在文件对话框里按住 Ctrl(macOS 是 ⌘)点选即可多选;没写名字的会按文件名补上。列表里是导入的技能和随应用附带的官方技能,每个都可以单独关闭。",
    en: "A skill is a markdown file of steps. The prompt carries only one line per skill (name + when); the body loads via use_skill only when needed — so skills cost almost no context. Put them in ~/.chaty/skills/ (global) or <project>/.chaty/skills/ (project, shadows global). \"Import skill files…\" copies one .md or several (another tool's SKILL.md, say) into ~/.chaty/skills/ — hold Ctrl (⌘ on macOS) in the file dialog to pick more than one — naming each from its file when it has no name. The list holds your imported skills and the ones bundled with Chaty; each can be turned off.", pt: "Uma skill é um arquivo markdown com passos. O prompt recebe apenas uma linha por skill (nome + uso); o corpo é carregado usando use_skill somente se houver necessidade — logo, skills quase não consomem contexto. Adicione-as em ~/.chaty/skills/ (global) ou <projeto>/.chaty/skills/ (projeto, substitui a global). \"Importar arquivos de skill…\" copia um ou vários .md (o SKILL.md de outra ferramenta, por exemplo) para ~/.chaty/skills/ — segure Ctrl (⌘ no macOS) na caixa de diálogo para escolher mais de um —, usando o nome do arquivo quando ele não tem nome. A lista mostra as skills importadas e as incluídas com o Chaty; cada uma pode ser desativada.",
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
  canvasErrMore: { zh: "等共 {n} 个错误", en: "— {n} errors total", pt: "— {n} erros no total" },
  canvasFixBtn: { zh: "修复", en: "Fix it", pt: "Conserte" },
  canvasIgnore: { zh: "忽略", en: "Ignore", pt: "Ignorar" },
  canvasMute: { zh: "本次不再提示", en: "Don't ask again", pt: "Não perguntar novamente" },
  canvasGenerating: { zh: "生成中…", en: "Generating…", pt: "Gerando…" },
  canvasNeedsModel: { zh: "请先加载模型", en: "Load a model first", pt: "Carregue um modelo primeiro" },
  canvasNoHtml: { zh: "未能从输出中解析出 HTML", en: "Couldn't parse HTML from the output", pt: "Não foi possível extrair o HTML da resposta do modelo" },
  ejectModel: { zh: "卸载模型（回到空状态）", en: "Eject model (back to empty)", pt: "Descarregar modelo (voltar ao início)" },
  deleteModelFile: { zh: "删除该模型文件", en: "Delete this model file", pt: "Excluir este arquivo de modelo" },
  confirmDeleteModel: {
    zh: "永久删除模型文件「{name}」？此操作不可撤销。",
    en: 'Permanently delete the model file "{name}"? This cannot be undone.', pt: 'Excluir permanentemente o arquivo de modelo "{name}"? Isso não pode ser desfeito.'
  },
  // html preview
  closePreview: { zh: "关闭预览", en: "Close preview", pt: "Fechar visualização" },
  imagePreview: { zh: "图片", en: "Image", pt: "Imagem" },
  noConversations: { zh: "暂无历史会话", en: "No conversations yet", pt: "Nenhuma conversa ainda" },
  deleteConv: { zh: "删除会话", en: "Delete conversation", pt: "Excluir conversa" },
  confirm: { zh: "确认", en: "Confirm", pt: "Confirmar" },
  confirmDelete: { zh: "删除", en: "Delete", pt: "Excluir" },
  confirmDeleteConv: {
    zh: "确定要删除这个对话吗？",
    en: "Delete this conversation?", pt: "Excluir esta conversa?"
  },
  confirmDeleteSession: {
    zh: "确定要删除这个编程会话吗？",
    en: "Delete this coding session?", pt: "Excluir esta sessão de código?"
  },
  pinConv: { zh: "置顶", en: "Pin", pt: "Fixar" },
  unpinConv: { zh: "取消置顶", en: "Unpin", pt: "Desafixar" },
  renameConv: { zh: "重命名", en: "Rename", pt: "Renomear" },
  // empty state
  readyMsg: { zh: "我已就绪——完全本地运行，对话不出本机。", en: "Ready — everything runs locally, nothing leaves your machine.", pt: "Pronto — tudo roda localmente, nada sai da sua máquina." },
  loadToStart: { zh: "加载一个本地模型，即可开始对话。", en: "Load a local model to start chatting.", pt: "Carregue um modelo local para iniciar o chat." },
  // composer
  inputPh: { zh: "输入消息，Enter 发送，Shift+Enter 换行", en: "Message… (Enter to send, Shift+Enter for newline)", pt: "Mensagem… (Enter para enviar, Shift+Enter para quebra de linha)" },
  inputPhMod: { zh: "输入消息，⌘/Ctrl+Enter 发送，Enter 换行", en: "Message… (⌘/Ctrl+Enter to send, Enter for newline)", pt: "Mensagem… (⌘/Ctrl+Enter para enviar, Enter para quebra de linha)" },
  inputPhWeb: { zh: "联网搜索已开启，输入问题…", en: "Web search on — ask anything…", pt: "Pesquisa web ativada — pergunte qualquer coisa…" },
  inputPhNoModel: { zh: "先加载一个模型…", en: "Load a model first…", pt: "Carregue um modelo primeiro…" },
  webDesignOff: { zh: "网页设计模式：已关闭（/webdesign 切换）", en: "Web design mode: off (/webdesign to toggle)", pt: "Modo web design: desligado (/webdesign para alternar)" },
  webDesignChip: { zh: "网页设计模式", en: "Web design mode", pt: "Modo web design" },
  inputPhDesign: { zh: "描述你想要的界面，模型会生成单文件 HTML…", en: "Describe the UI you want — get a single-file HTML…", pt: "Descreva a interface que você deseja — receba um HTML de arquivo único…" },
  toolsMenu: { zh: "工具", en: "Tools", pt: "Ferramentas" },
  toolAttach: { zh: "添加附件", en: "Attach a file", pt: "Anexar arquivo" },
  toolWeb: { zh: "联网搜索", en: "Web search", pt: "Pesquisa na web" },
  toolThink: { zh: "思考模式", en: "Thinking mode", pt: "Modo de raciocínio" },
  toolDesign: { zh: "网页设计模式", en: "Web design mode", pt: "Modo web design" },
  thinkUnsupported: { zh: "当前模型不支持思考模式", en: "This model doesn't support thinking", pt: "Este modelo não suporta modo de raciocínio" },
  // update banner
  updateAvailable: { zh: "发现新版本 v{v}", en: "Update available — v{v}", pt: "Atualização disponível — v{v}" },
  updateNow: { zh: "立即更新", en: "Update now", pt: "Atualizar agora" },
  updateLater: { zh: "稍后", en: "Later", pt: "Mais tarde" },
  updateDownloading: { zh: "下载中…", en: "Downloading…", pt: "Baixando…" },
  // context usage
  ctxUsage: { zh: "上下文占用", en: "Context usage", pt: "Uso de contexto" },
  stopTitle: { zh: "停止生成", en: "Stop", pt: "Parar" },
  sendTitle: { zh: "发送", en: "Send", pt: "Enviar" },
  micStart: { zh: "语音输入", en: "Voice input", pt: "Entrada de voz" },
  micStop: { zh: "停止录音", en: "Stop recording", pt: "Parar gravação" },
  voiceDlStt: {
    zh: "正在下载语音识别模型(仅首次使用需要)… {progress}",
    en: "Downloading the speech recognition model (first use only)… {progress}",
    pt: "Baixando o modelo de reconhecimento de fala (só no primeiro uso)… {progress}",
  },
  voiceDlTts: {
    zh: "正在下载语音合成模型(仅首次使用需要)… {progress}",
    en: "Downloading the speech synthesis model (first use only)… {progress}",
    pt: "Baixando o modelo de síntese de fala (só no primeiro uso)… {progress}",
  },
  liveStart: { zh: "实时语音对话", en: "Live voice chat", pt: "Chat de voz ao vivo" },
  liveExit: { zh: "退出实时模式", en: "Exit live mode", pt: "Sair do modo ao vivo" },
  liveListening: { zh: "聆听中…", en: "Listening…", pt: "Ouvindo…" },
  liveThinking: { zh: "思考中…", en: "Thinking…", pt: "Pensando…" },
  liveSpeaking: { zh: "回答中…", en: "Speaking…", pt: "Falando…" },
  toolVoiceGroup: { zh: "语音回复", en: "Voice replies", pt: "Respostas por voz" },
  speakAloud: { zh: "朗读回复", en: "Read replies aloud", pt: "Ler respostas em voz alta" },
  // context menu
  ctxCut: { zh: "剪切", en: "Cut", pt: "Recortar" },
  ctxCopy: { zh: "复制", en: "Copy", pt: "Copiar" },
  ctxCopyMessage: { zh: "复制整条消息", en: "Copy message", pt: "Copiar mensagem" },
  ctxPaste: { zh: "粘贴", en: "Paste", pt: "Colar" },
  ctxSelectAll: { zh: "全选", en: "Select all", pt: "Selecionar tudo" },
  // window controls
  minimize: { zh: "最小化", en: "Minimize", pt: "Minimizar" },
  maximize: { zh: "最大化", en: "Maximize", pt: "Maximizar" },
  restore: { zh: "向下还原", en: "Restore", pt: "Restaurar" },
  close: { zh: "关闭", en: "Close", pt: "Fechar" },
  // message actions
  sources: { zh: "来源", en: "Sources", pt: "Fontes" },
  copy: { zh: "复制", en: "Copy", pt: "Copiar" },
  fork: { zh: "分叉", en: "Branch", pt: "Bifurcar (Branch)" },
  copyTitle: { zh: "复制回答", en: "Copy reply", pt: "Copiar resposta" },
  forkTitle: { zh: "从这里分叉为新对话", en: "Branch into a new chat from here", pt: "Criar um novo chat a partir daqui" },
  reread: { zh: "朗读", en: "Replay", pt: "Ouvir novamente" },
  rereadStop: { zh: "停止", en: "Stop", pt: "Parar" },
  rereadTitle: { zh: "朗读这条回复", en: "Read this reply aloud", pt: "Ler esta resposta em voz alta" },
  regenerate: { zh: "重新生成", en: "Regenerate", pt: "Regerar" },
  regenTitle: { zh: "重新生成这条回答（会丢弃其后的对话）", en: "Regenerate this reply (drops what follows)", pt: "Regerar esta resposta (descarta as seguintes)" },
  editMsg: { zh: "编辑", en: "Edit", pt: "Editar" },
  saveEdit: { zh: "保存并重发", en: "Save & resend", pt: "Salvar e reenviar" },
  cancel: { zh: "取消", en: "Cancel", pt: "Cancelar" },
  // assistant message
  searching: { zh: "正在联网搜索…", en: "Searching the web…", pt: "Pesquisando na web…" },
  searchingKb: { zh: "正在检索知识库…", en: "Searching the knowledge base…", pt: "Pesquisando na base de conhecimento…" },
  searchingMix: { zh: "正在检索知识库与网络…", en: "Searching knowledge base & web…", pt: "Pesquisando base de conhecimento e web…" },
  composing: { zh: "正在整理上文…", en: "Composing earlier context…", pt: "Organizando contexto anterior…" },
  contextSummary: {
    zh: "【早前对话摘要，供延续参考】\n",
    en: "[Summary of earlier conversation, for continuity]\n", pt: "[Resumo da conversa anterior, para manter a continuidade]\n"
  },
  // A turn can end with nothing to show. Say which way it happened rather than
  // leaving an empty bubble that reads as the app losing the answer.
  emptyNoRoom: {
    zh: "**(这一轮没有生成内容:提示词已超出模型的上下文窗口。请开启新对话,或在「设置 → 模型 → 上下文长度」里调大。)**",
    en: "**(Nothing was generated this turn: the prompt no longer fits the model's context window. Start a new conversation, or raise Settings → Model → Context length.)**", pt: "**(Nenhum conteúdo gerado neste turno: o prompt não cabe mais na janela de contexto do modelo. Inicie uma nova conversa ou aumente em Configurações → Modelo → Tamanho do contexto.)**"
  },
  emptyOutOfBudget: {
    zh: "**(这一轮的生成长度用完时,模型还停在思考里,没来得及写出回答。可在「设置 → 采样 → 最大生成长度」里调大,或点击重新生成。)**",
    en: "**(The model was still reasoning when it ran out of generation length, so no answer was written. Raise Settings → Sampling → Max length, or regenerate.)**", pt: "**(O modelo ainda estava pensando quando o limite de geração esgotou, então nenhuma resposta foi escrita. Aumente em Configurações → Amostragem → Comprimento máximo, ou regere.)**"
  },
  emptyThoughtOnly: {
    zh: "**(模型这一轮只输出了思考过程就结束了,没有给出正式回答。可以点击重新生成。)**",
    en: "**(The model spent this turn reasoning and stopped without writing an answer. Regenerating usually gets one.)**", pt: "**(O modelo gastou este turno apenas raciocinando e parou antes de escrever uma resposta. Regerar geralmente resolve.)**"
  },
  thinking: { zh: "正在思考", en: "Thinking", pt: "Pensando" },
  thoughtExpand: { zh: "已深度思考 · 点击展开", en: "Reasoned · click to expand", pt: "Raciocinou · clique para expandir" },
  thoughtCollapse: { zh: "已深度思考 · 点击收起", en: "Reasoned · click to collapse", pt: "Raciocinou · clique para recolher" },
  // attachment
  removeAttach: { zh: "移除附件", en: "Remove attachment", pt: "Remover anexo" },
  visionAttach: { zh: "图片 · 模型将直接查看", en: "Image · the model will see it", pt: "Imagem · o modelo irá vê-la" },
  cmAttachFile: { zh: "附加文件(文档或图片)", en: "Attach a file (document or image)", pt: "Anexar um arquivo (documento ou imagem)" },
  cmClickPreview: { zh: "点击预览", en: "click to preview", pt: "clique para visualizar" },
  cmSaveImage: { zh: "保存到本地", en: "Save to disk", pt: "Salvar no disco" },
  cmSaved: { zh: "已保存", en: "Saved", pt: "Salvo" },
  cmScreenshot: { zh: "网页截图", en: "Screenshot", pt: "Captura de tela" },
  attachContextLabel: { zh: "附件", en: "Attachment", pt: "Anexo" },
  cmRefSession: { zh: "引用会话", en: "Referenced session", pt: "Sessão referenciada" },
  cmRefRemove: { zh: "取消引用", en: "Remove the reference", pt: "Remover a referência" },
  cmRefHint: {
    zh: "要看这个会话更多内容,用 search_history,session 写这个 id。",
    en: "For more of it, call search_history with session set to that id.",
    pt: "Para ver mais, chame search_history com session igual a esse id.",
  },
  visionBadge: { zh: "视觉", en: "Vision", pt: "Visão" },
  visionBadgeTip: { zh: "支持视觉——加载后可直接理解图片", en: "Vision-capable — understands images once loaded", pt: "Capaz de visão — entende imagens uma vez carregadas" },
  mlxBadgeTip: { zh: "MLX 文件夹模型，由 Apple Silicon 专用引擎运行", en: "MLX folder model — runs on the Apple-Silicon-native engine", pt: "Modelo em pasta MLX — roda no motor nativo Apple-Silicon" },
  storeSearchPh: { zh: "搜索模型，或粘贴仓库链接后回车…", en: "Search models, or paste a repo link and press Enter…", pt: "Pesquisar modelos, ou colar link de repo e pressionar Enter…" },
  storeFormat: { zh: "格式", en: "Format", pt: "Formato" },
  storeSort: { zh: "排序", en: "Sort", pt: "Ordenar" },
  storeTrending: { zh: "热门", en: "Trending", pt: "Em alta" },
  storeDownloads: { zh: "下载量", en: "Downloads", pt: "Downloads" },
  storeLikes: { zh: "点赞", en: "Likes", pt: "Curtidas" },
  storeUpdated: { zh: "最近更新", en: "Recently updated", pt: "Atualizado recentemente" },
  storeNoResults: { zh: "没有找到模型", en: "No models found", pt: "Nenhum modelo encontrado" },
  storePickModel: { zh: "从左侧选择一个模型查看详情", en: "Pick a model on the left to see details", pt: "Escolha um modelo à esquerda para ver detalhes" },
  storeFitsRam: { zh: "✓ 可完整载入内存", en: "✓ Fits fully in memory", pt: "✓ Cabe totalmente na memória" },
  storeOverRam: { zh: "≈ 接近或超出内存，速度可能受影响", en: "≈ Near or over memory — may run slowly", pt: "≈ Próximo ou acima da memória — pode rodar lento" },
  storeVisionIncluded: { zh: "自动附带视觉编码器", en: "vision encoder included", pt: "codificador de visão incluído" },
  storeReadmeEmpty: { zh: "该仓库没有 README", en: "This repo has no README", pt: "Este repositório não possui README" },
  storeToday: { zh: "今天", en: "today", pt: "hoje" },
  storeDaysAgo: { zh: "{n} 天前", en: "{n} days ago", pt: "Há {n} dias" },
  storeMlxMacOnly: { zh: "MLX 模型仅支持 macOS (Apple Silicon)——请选择 GGUF 版本", en: "MLX models are macOS (Apple Silicon) only — pick a GGUF build instead", pt: "Modelos MLX são apenas para macOS (Apple Silicon) — escolha uma build GGUF" },
  mmprojFailed: {
    zh: "视觉编码器加载失败，本次会话仅支持文本（图片将走 OCR）。",
    en: "The vision encoder failed to load — text only this session (images fall back to OCR).", pt: "Falha ao carregar o codificador de visão — apenas texto nesta sessão (imagens caem para OCR)."
  },
  miVision: { zh: "视觉", en: "Vision", pt: "Visão" },
  setupVision: { zh: "支持看图", en: "Understands images", pt: "Entende imagens" },
  truncatedSuffix: { zh: " · 已截断", en: " · truncated", pt: " · truncado" },
  charsLabel: { zh: "{n} 字", en: "{n} chars", pt: "{n} caracteres" },
  readAttachFailed: { zh: "读取附件失败", en: "Failed to read file", pt: "Falha ao ler o arquivo" },
  dropToAttach: { zh: "松开以添加为附件", en: "Drop to attach", pt: "Solte para anexar" },
  searchConv: { zh: "搜索对话…", en: "Search chats…", pt: "Pesquisar conversas…" },
  noMatches: { zh: "无匹配对话", en: "No matching chats", pt: "Nenhuma conversa correspondente" },
  exportTitle: { zh: "导出对话", en: "Export chat", pt: "Exportar conversa" },
  exportMd: { zh: "导出为 Markdown", en: "Export as Markdown", pt: "Exportar como Markdown" },
  exportJson: { zh: "导出为 JSON", en: "Export as JSON", pt: "Exportar como JSON" },
  exportFailed: { zh: "导出失败", en: "Export failed", pt: "Falha na exportação" },
  dlTitle: { zh: "下载模型", en: "Download model", pt: "Baixar modelo" },
  dlHint: {
    zh: "输入 HuggingFace 仓库（owner/name）或其网址，列出其中的模型文件（GGUF / MLX）后下载到模型文件夹。",
    en: "Enter a HuggingFace repo (owner/name) or URL to list its model files (GGUF / MLX) and download into your models folder.", pt: "Insira um repositório HuggingFace (owner/name) ou URL para listar os arquivos (GGUF / MLX) e baixar para sua pasta de modelos."
  },
  dlGet: { zh: "下载", en: "Download", pt: "Baixar" },
  dlSearchFailed: { zh: "查找失败", en: "Lookup failed", pt: "Falha na busca" },
  dlFailed: { zh: "下载失败", en: "Download failed", pt: "Falha no download" },
  // settings
  settingsTitle: { zh: "设置", en: "Settings", pt: "Configurações" },
  language: { zh: "语言", en: "Language", pt: "Idioma" },
  systemPrompt: { zh: "系统提示词", en: "System prompt", pt: "Prompt de sistema"  },
  systemPromptPh: { zh: "可选，例如：你是一个简洁、专业的助手。", en: "Optional, e.g. You are a concise, professional assistant.", pt: "Opcional, ex: Você é um assistente conciso e profissional." },
  temperature: { zh: "温度", en: "Temperature", pt: "Temperatura" },
  maxTokens: { zh: "最大生成长度", en: "Max length", pt: "Comprimento máximo" },
  off: { zh: "关", en: "off", pt: "desligado" },
  repeatPenalty: { zh: "重复惩罚", en: "Repeat penalty", pt: "Penalidade de repetição" },
  stopSeqs: { zh: "停止词（每行一个）", en: "Stop sequences (one per line)", pt: "Sequências de parada (uma por linha)" },
  stopSeqsPh: { zh: "例如：\nUser:\n###", en: "e.g.\nUser:\n###", pt: "ex:\\nUser:\\n###" },
  presets: { zh: "提示词预设", en: "Prompt presets", pt: "Predefinições de prompt" },
  presetNamePh: { zh: "预设名称…", en: "Preset name…", pt: "Nome da predefinição…" },
  presetSave: { zh: "保存当前", en: "Save", pt: "Salvar" },
  voice: { zh: "语音发音人", en: "Voice", pt: "Voz" },
  voiceSpeed: { zh: "语速", en: "Speech rate", pt: "Velocidade da voz" },
  theme: { zh: "主题", en: "Theme", pt: "Tema" },
  themeSystem: { zh: "跟随系统", en: "System", pt: "Sistema" },
  themeLight: { zh: "浅色", en: "Light", pt: "Claro" },
  themeDark: { zh: "深色", en: "Dark", pt: "Escuro" },
  resetDefaults: { zh: "恢复默认", en: "Reset", pt: "Restaurar padrões" },
  // GPU acceleration settings
  gpuAccel: { zh: "GPU 加速", en: "GPU acceleration", pt: "Aceleração por GPU" },
  gpuAuto: { zh: "自动", en: "Auto", pt: "Automático" },
  gpuOff: { zh: "关闭", en: "Off", pt: "Desligado" },
  gpuCustom: { zh: "自定义", en: "Custom", pt: "Personalizado" },
  gpuLayersLabel: { zh: "GPU 层数", en: "GPU layers", pt: "Camadas na GPU" },
  specDecode: { zh: "推测解码", en: "Speculative decoding", pt: "Decodificação especulativa" },
  experimental: { zh: "实验性", en: "experimental", pt: "experimental" },
  specDecodeHint: {
    zh: "用模型自带的预测头先猜出接下来的几个词，再让模型一次性校验，回答内容不变、只是更快。加速幅度随模型和内容而定，更改将在下次加载模型时生效。",
    en: "The model's own prediction head guesses the next few tokens and the model checks them in one pass — same reply, fewer passes. How much it helps depends on the model and on what is being written. Changes apply on the next model load.",
  },
  specDecodeUnsupported: {
    zh: "当前模型不带预测头，无法使用。",
    en: "The loaded model carries no prediction head, so there is nothing to speculate with.",
  },
  gpuHint: {
    zh: "自动模式会按显存把尽量多的层放到 GPU。更改将在下次加载模型时生效。",
    en: "Auto fills the GPU with as many layers as VRAM allows. Changes apply on the next model load.", pt: "Auto preenche a GPU com o máximo de camadas que a VRAM permitir. As alterações se aplicam no próximo carregamento de modelo."
  },
  // context length settings
  ctxLength: { zh: "上下文长度", en: "Context length", pt: "Tamanho do contexto" },
  ctxAuto: { zh: "自动", en: "Auto", pt: "Automático" },
  ctxTokens: { zh: "上下文 Token", en: "Context tokens", pt: "Tokens de contexto" },
  ctxHint: {
    zh: "自动 = 在内存允许范围内尽量用满模型的训练上下文。自定义值也会按内存上限自动回落。",
    en: "Auto = as much of the model's trained context as memory allows. Custom values are also capped to fit memory.", pt: "Auto = o máximo de contexto treinado do modelo que a memória permitir. Valores personalizados também são limitados para caber na memória."
  },
  reloadApply: { zh: "重新加载模型以生效", en: "Reload model to apply", pt: "Recarregar modelo para aplicar" },
  // data management
  openDataDir: { zh: "打开数据文件夹", en: "Open data folder", pt: "Abrir pasta de dados" },
  clearAllChats: { zh: "清空所有对话", en: "Clear all chats", pt: "Limpar todas as conversas" },
  confirmClearChats: {
    zh: "确定要删除全部对话记录吗？此操作不可撤销。",
    en: "Delete all conversations? This cannot be undone.", pt: "Excluir todas as conversas? Isso não pode ser desfeito."
  },
  dataHint: {
    zh: "所有对话、模型与索引都保存在本地数据文件夹中，可手动复制备份。",
    en: "All conversations, models and indexes live in the local data folder — copy it to back up.", pt: "Todas as conversas, modelos e índices ficam na pasta local de dados — copie-a para fazer backup."
  },
  dataFolder: { zh: "数据文件夹", en: "Data folder", pt: "Pasta de dados" },
  clearChatsHint: {
    zh: "删除全部对话与 Code 会话记录，不可撤销。",
    en: "Deletes every conversation and Code session. Irreversible.", pt: "Exclui todas as conversas e sessões de Código. Irreversível."
  },
  clearKb: { zh: "清空知识库", en: "Clear knowledge base", pt: "Limpar base de conhecimento" },
  clearKbHint: {
    zh: "移除全部已索引文档与向量数据，原始文件不受影响。",
    en: "Removes all indexed documents and vectors; original files are untouched.", pt: "Remove todos os documentos indexados e vetores; os arquivos originais permanecem intocados."
  },
  clearKbConfirm: {
    zh: "确定要清空知识库吗？所有已索引的文档都将被移除。",
    en: "Clear the knowledge base? All indexed documents will be removed.", pt: "Limpar a base de conhecimento? Todos os documentos indexados serão removidos."
  },
  statConvs: { zh: "对话", en: "Conversations", pt: "Conversas" },
  statMsgs: { zh: "消息", en: "Messages", pt: "Mensagens" },
  statCodeSessions: { zh: "Code 会话", en: "Code sessions", pt: "Sessões de código" },
  statModels: { zh: "本地模型", en: "Local models", pt: "Modelos locais" },
  statKbDocs: { zh: "知识库文档", en: "KB documents", pt: "Documentos (KB)" },
  statKbChunks: { zh: "知识块", en: "KB chunks", pt: "Fragmentos (KB)" },
  statDbSize: { zh: "数据库占用", en: "Database size", pt: "Tamanho do banco de dados" },
  // general / chat / model preferences
  setDarkScheme: { zh: "深色方案", en: "Dark palette", pt: "Paleta escura" },
  setDarkSchemeHint: { zh: "深色模式使用的配色", en: "Colours used in dark mode", pt: "Cores usadas no modo escuro" },
  setLightScheme: { zh: "浅色方案", en: "Light palette", pt: "Paleta clara" },
  setLightSchemeHint: { zh: "浅色模式使用的配色", en: "Colours used in light mode", pt: "Cores usadas no modo claro" },
  schemeWarmCharcoal: { zh: "暖炭", en: "Warm charcoal", pt: "Carvão quente" },
  schemeCoolCharcoal: { zh: "冷黑", en: "Cool charcoal", pt: "Carvão frio" },
  schemePaper: { zh: "纸白", en: "Paper", pt: "Papel" },
  schemeCream: { zh: "暖米", en: "Cream", pt: "Creme" },
  setCodeTheme: { zh: "代码高亮", en: "Code highlighting", pt: "Destaque de código" },
  setCodeThemeHint: { zh: "回答中代码块的配色", en: "Palette for code blocks in answers", pt: "Paleta para os blocos de código nas respostas" },
  setUiScale: { zh: "界面缩放", en: "UI scale", pt: "Escala da interface" },
  setUiScaleHint: { zh: "整体放大或缩小界面", en: "Zoom the whole interface", pt: "Dá zoom em toda a interface" },
  setSendKey: { zh: "发送快捷键", en: "Send shortcut", pt: "Atalho de envio" },
  setSendKeyHint: {
    zh: "选择组合键时，Enter 换行",
    en: "With the combo, Enter inserts a newline", pt: "Com a combinação, Enter insere uma quebra de linha"
  },
  setReduceMotion: { zh: "减少动效", en: "Reduce motion", pt: "Reduzir movimento" },
  setReduceMotionHint: {
    zh: "关闭界面过渡与弹出动画",
    en: "Disable UI transitions and pop-in animations", pt: "Desativa transições de interface e animações de pop-in"
  },
  setAnswerSize: { zh: "回答字号", en: "Answer text size", pt: "Tamanho do texto de resposta" },
  setAnswerSizeHint: { zh: "模型回答的阅读字号", en: "Reading size for model answers", pt: "Tamanho de leitura para as respostas do modelo" },
  sizeSm: { zh: "紧凑", en: "Compact", pt: "Compacto" },
  sizeMd: { zh: "标准", en: "Standard", pt: "Padrão" },
  sizeLg: { zh: "舒适", en: "Comfortable", pt: "Confortável" },
  setAutoTitle: { zh: "自动命名对话", en: "Auto-name conversations", pt: "Nomear conversas automaticamente" },
  setAutoTitleHint: {
    zh: "首轮回复后自动生成标题",
    en: "Generate a title after the first reply", pt: "Gera um título após a primeira resposta"
  },
  setAutoLoadLast: { zh: "启动时加载上次模型", en: "Load last model on launch", pt: "Carregar último modelo ao iniciar" },
  setAutoLoadLastHint: {
    zh: "打开应用即恢复上次使用的模型",
    en: "Restore the previously used model at startup", pt: "Restaura o modelo utilizado anteriormente ao iniciar"
  },
  modelsFolder: { zh: "模型文件夹", en: "Models folder", pt: "Pasta de modelos" },
  modelsFolderHint: {
    zh: "放入 GGUF 文件或 MLX 模型文件夹即可被识别。更改位置后只读取新目录",
    en: "Drop GGUF files or MLX model folders here to make them available. Changing the location makes it the only folder read", pt: "Arraste arquivos GGUF ou pastas de modelo MLX aqui para disponibilizá-los. Alterar o local faz dele a única pasta lida"
  },
  modelsDirHintToast: {
    zh: "将模型文件放入此文件夹后，重新打开模型选择器即可看到",
    en: "Drop model files into this folder, then reopen the model picker to see them", pt: "Arraste arquivos de modelo para esta pasta, depois reabra o seletor de modelos para vê-los"
  },
  hfEndpoint: { zh: "HuggingFace 端点", en: "HuggingFace endpoint", pt: "Endpoint HuggingFace" },
  hfEndpointOfficial: { zh: "官方", en: "Official", pt: "Oficial" },
  hfEndpointCustom: { zh: "自定义", en: "Custom", pt: "Personalizado" },
  hfEndpointHint: {
    zh: "模型搜索、下载和知识库模型都走此端点。大陆网络推荐 hf-mirror.com；镜像不支持 xet 回退，个别文件失败时请切回官方。",
    en: "Model search, downloads and the knowledge-base model all use this endpoint. hf-mirror.com is recommended in mainland China; mirrors lack the xet fallback — switch back to Official if a file fails.", pt: "Busca de modelos, downloads e modelo de base de conhecimento utilizam este endpoint. hf-mirror.com é recomendado na China continental; mirrors não têm o fallback xet — volte para Oficial se um arquivo falhar."
  },
  tipHfEndpoint: {
    zh: "huggingface.co 不可达时可切换到与其路径兼容的镜像站",
    en: "Switch to a path-compatible mirror when huggingface.co is unreachable", pt: "Mude para um mirror compatível com o caminho quando huggingface.co estiver inacessível"
  },
  voicePreview: { zh: "试听", en: "Preview", pt: "Prévia" },
  voicePreviewHint: {
    zh: "用当前语音和语速播放示例",
    en: "Play a sample with the current voice and speed", pt: "Tocar um exemplo com a voz e velocidade atuais"
  },
  voicePreviewBtn: { zh: "播放示例", en: "Play sample", pt: "Tocar exemplo" },
  chineseVoice: { zh: "中文语音支持", en: "Chinese voice support", pt: "Suporte a voz em chinês" },
  chineseVoiceHint: {
    zh: "识别用多语言 Whisper,含中文的回复由中文 VITS 朗读。中文界面默认开启,关掉即回到纯英文的 base.en。",
    en: "Recognition uses multilingual Whisper, and a reply containing Chinese is read by the Chinese VITS voice. On by default in the Chinese interface; turn it off to keep English-only base.en and its accuracy.", pt: "O reconhecimento usa o Whisper multilíngue, e as respostas que contêm chinês são lidas pela voz VITS em chinês. Ativado por padrão na interface chinesa; desligue para priorizar o modelo em inglês base.en e garantir a precisão."
  },
  voiceEn: { zh: "英文音色", en: "English voice", pt: "Voz em inglês" },
  voicePreviewZh: { zh: "播放中文示例", en: "Play Chinese sample", pt: "Tocar exemplo em chinês" },
  voicePreviewEn: { zh: "播放英文示例", en: "Play English sample", pt: "Tocar exemplo em inglês" },
  voiceZh: { zh: "中文音色", en: "Chinese voice", pt: "Voz em chinês" },
  voiceZhHint: {
    zh: "中文回复用哪个说话人。与上面的英文音色各自独立,互不影响。",
    en: "Which speaker reads Chinese replies. Kept apart from the English voice above — changing one never moves the other.", pt: "Qual apresentador fará a leitura em chinês. Mantido separado da voz em inglês acima — mudar um não altera o outro."
  },
  voiceEngineHint: {
    zh: "语音完全在本地运行：多语言 Whisper 自动识别中文和英文，中文回复由 VITS 朗读，英文回复由 Kokoro 朗读。首次使用对应语言时会下载模型。",
    en: "Voice runs locally. English uses Whisper base.en and Kokoro by default; Chinese support adds multilingual Whisper and VITS downloads.", pt: "A voz roda localmente. O inglês usa o Whisper base.en e o Kokoro por padrão; o suporte ao chinês requer baixar o Whisper multilíngue e VITS."
  },
  // knowledge base (local RAG)
  kbTitle: { zh: "本地知识库", en: "Local knowledge base", pt: "Base de conhecimento local" },
  kbDocs: { zh: "个文档", en: "docs", pt: "documentos" },
  kbChunks: { zh: "个片段", en: "chunks", pt: "fragmentos" },
  kbModelNote: {
    zh: "首次使用需下载多语嵌入模型 bge-m3（约 730 MB，一次性，之后完全离线）。",
    en: "First use downloads the multilingual bge-m3 embedding model (~730 MB, one-time; fully offline after).", pt: "O primeiro uso baixa o modelo multilíngue de embeddings bge-m3 (~730 MB, uma única vez; totalmente offline depois disso)."
  },
  kbDownloadModel: { zh: "下载嵌入模型", en: "Download embedding model", pt: "Baixar modelo de embeddings" },
  kbEmpty: { zh: "还没有文档 — 添加 PDF / 文本 / 图片开始", en: "No documents yet — add a PDF / text / image file", pt: "Nenhum documento ainda — adicione um arquivo PDF / texto / imagem" },
  kbAdd: { zh: "添加文档", en: "Add documents", pt: "Adicionar documentos" },
  kbAddFolder: { zh: "导入文件夹", en: "Import folder", pt: "Importar pasta" },
  kbFolderEmpty: {
    zh: "该文件夹（含子目录）里没有可导入的文件。",
    en: "No importable files were found in that folder (or its subfolders).", pt: "Nenhum arquivo importável foi encontrado nessa pasta (ou em suas subpastas)."
  },
  kbFolderConfirm: {
    zh: "在该文件夹及子目录中找到 {n} 个文件，全部导入知识库？",
    en: "Found {n} files in that folder and its subfolders. Import all of them?", pt: "Encontrados {n} arquivos nessa pasta e subpastas. Importar todos?"
  },
  kbFolderConfirmGo: { zh: "全部导入", en: "Import all", pt: "Importar tudo" },
  kbScopeTip: { zh: "勾选 = 参与检索", en: "Checked = included in retrieval", pt: "Marcado = incluído na busca" },
  kbScopeHint: {
    zh: "勾选要检索的文档，未勾选的不参与回答。",
    en: "Check the documents to search; unchecked ones are excluded from answers.", pt: "Marque os documentos a pesquisar; desmarcados são excluídos das respostas."
  },
  kbRemove: { zh: "移除", en: "Remove", pt: "Remover" },
  kbIndexing: { zh: "索引中", en: "Indexing", pt: "Indexando" },
  // deep-dive podcast (NotebookLM-style)
  kbPodcast: { zh: "生成深度播客", en: "Generate deep-dive podcast", pt: "Gerar podcast aprofundado" },
  kbReport: { zh: "生成报告", en: "Generate report", pt: "Gerar relatório" },
  kbClear: { zh: "清空知识库", en: "Clear all", pt: "Limpar tudo" },
  kbClearConfirm: {
    zh: "确定要清空整个知识库吗？所有已索引的文档都会被移除（嵌入模型保留）。此操作无法撤销。",
    en: "Clear the entire knowledge base? All indexed documents will be removed (the embedding model is kept). This can't be undone.", pt: "Limpar toda a base de conhecimento? Todos os documentos indexados serão removidos (o modelo de embeddings é mantido). Isso não pode ser desfeito."
  },
  kbReportTitle: { zh: "知识库报告", en: "Knowledge-base report", pt: "Relatório da base de conhecimento" },
  kbReportEmpty: {
    zh: "正在根据你的知识库内容生成概览报告……",
    en: "Generating an overview report from your knowledge base…", pt: "Gerando um relatório geral a partir da sua base de conhecimento…"
  },
  kbReportRunning: {
    zh: "正在根据知识库内容智能生成报告（按文件引用，全程不联网）",
    en: "Generating a report from your knowledge base (cited per file, fully offline)", pt: "Gerando relatório da base de conhecimento (citado por arquivo, totalmente offline)"
  },
  kbReportRegen: { zh: "重新生成", en: "Regenerate", pt: "Regerar" },
  kbReportFiles: { zh: "个文件", en: "files", pt: "arquivos" },
  kbReportPhasePlan: { zh: "正在梳理知识库…", en: "Reviewing the knowledge base…", pt: "Revisando a base de conhecimento…" },
  kbReportPhaseRead: { zh: "正在读取文档内容…", en: "Reading the documents…", pt: "Lendo os documentos…" },
  kbReportNeedModel: {
    zh: "请先加载一个对话模型，再生成知识库报告。",
    en: "Load a chat model first, then generate the knowledge-base report.", pt: "Carregue um modelo de chat primeiro, depois gere o relatório da base de conhecimento."
  },
  podcastTitle: { zh: "深度播客", en: "Deep-dive podcast", pt: "Podcast aprofundado" },
  podcastSub: {
    zh: "双主持人 · 英文 · 基于本地知识库",
    en: "Two hosts · English · from your knowledge base", pt: "Dois apresentadores · Inglês · a partir da base de conhecimento"
  },
  podcastIntro: {
    zh: "根据你启用的知识库文档，由模型撰写一段英文双人对话脚本，再用 Kokoro 一男一女两种音色交替朗读。生成期间其他 LLM 功能会暂时锁定，可随时取消，完成后可播放并导出音频。（仅生成英文播客）",
    en: "From your enabled documents, the model writes a two-host English script, then Kokoro reads it aloud with alternating male/female voices. Other LLM features are locked while it runs; you can cancel anytime, then play and export the audio. (English podcast only.)", pt: "A partir dos documentos ativados, o modelo escreve um roteiro em inglês para dois apresentadores, e o Kokoro faz a leitura em voz alta alternando vozes masculinas e femininas. As demais funções do LLM ficam bloqueadas enquanto roda; você pode cancelar a qualquer momento, e depois reproduzir ou exportar o áudio. (Podcast apenas em inglês.)"
  },
  podcastStart: { zh: "开始生成", en: "Generate", pt: "Gerar" },
  podcastWriting: { zh: "正在撰写脚本…", en: "Writing the script…", pt: "Escrevendo o roteiro…" },
  podcastVoicing: { zh: "正在合成语音…", en: "Synthesizing voices…", pt: "Sintetizando as vozes…" },
  podcastEta: { zh: "预计剩余", en: "Time left", pt: "Tempo restante" },
  etaLeft: { zh: "预计剩余", en: "Time left", pt: "Tempo restante" },
  podcastCancel: { zh: "取消生成", en: "Cancel", pt: "Cancelar" },
  podcastTurns: { zh: "段对话", en: "turns", pt: "turnos" },
  podcastPlay: { zh: "播放", en: "Play", pt: "Reproduzir" },
  podcastStop: { zh: "停止", en: "Stop", pt: "Parar" },
  podcastExport: { zh: "导出音频", en: "Export audio", pt: "Exportar áudio" },
  podcastRegen: { zh: "重新生成", en: "Regenerate", pt: "Regerar" },
  podcastRetry: { zh: "重试", en: "Try again", pt: "Tentar novamente" },
  podcastNeedModel: { zh: "请先加载一个聊天模型", en: "Load a chat model first", pt: "Carregue um modelo de chat primeiro" },
  podcastNoScript: {
    zh: "脚本生成失败，请重试或更换模型",
    en: "Could not produce a usable script — try again or switch models", pt: "Não foi possível produzir um roteiro utilizável — tente novamente ou mude de modelo" 
  },
  // Rendered only for zh UI users (the panel gates on lang) — the en text
  // exists to keep the "en is the universal fallback" invariant airtight.
  podcastFootZh: {
    zh: "提示：播客内容为英文，适合用于英语听力与学习。",
    en: "Note: podcast output is in English.", pt: "Nota: a saída do podcast é em inglês."
  },
  toolKb: { zh: "知识库检索", en: "Knowledge base", pt: "Base de conhecimento" },
  toolKbManage: { zh: "管理知识库…", en: "Manage knowledge base…", pt: "Gerenciar base de conhecimento…" },
  kbNeedSetup: { zh: "请先在知识库面板下载嵌入模型并添加文档", en: "Download the embedding model and add documents first", pt: "Baixe o modelo de embeddings e adicione documentos primeiro" },
  // tools-menu groups (hover submenus)
  toolKbGroup: { zh: "知识库", en: "Knowledge base", pt: "Base de conhecimento" },
  toolWebGroup: { zh: "联网功能", en: "Web", pt: "Web" },
  // deep research
  toolDeepResearch: { zh: "深度研究", en: "Deep Research", pt: "Pesquisa Profunda (Deep Research)" },
  drTitle: { zh: "深度研究", en: "Deep Research", pt: "Pesquisa Profunda" },
  drTopicPh: {
    zh: "输入研究主题，例如：固态电池的最新进展与商业化前景",
    en: "Enter a topic, e.g. the state of solid-state batteries and commercialization outlook", pt: "Insira um tópico, ex: o estado atual das baterias de estado sólido e perspectivas de mercado"
  },
  drDepth: { zh: "深度", en: "Depth", pt: "Profundidade" },
  drDepthQuick: { zh: "快速（2 轮）", en: "Quick (2 rounds)", pt: "Rápida (2 rodadas)" },
  drDepthStd: { zh: "标准（3 轮）", en: "Standard (3 rounds)", pt: "Padrão (3 rodadas)" },
  drDepthDeep: { zh: "深入（4 轮）", en: "Deep (4 rounds)", pt: "Profunda (4 rodadas)" },
  drRun: { zh: "开始研究", en: "Research", pt: "Pesquisar" },
  drStop: { zh: "停止", en: "Stop", pt: "Parar" },
  drEmpty: {
    zh: "给出一个主题，模型会自动多轮检索网络、边查边推理，最后写成一篇带参考来源的深度报告，可导出 PDF。",
    en: "Give a topic; the model searches the web over several rounds, reasons as it goes, and writes a cited in-depth report you can export to PDF.", pt: "Informe um tópico; o modelo fará buscas na web em várias rodadas, raciocinará durante o processo e redigirá um relatório aprofundado com citações, que você poderá exportar para PDF."
  },
  drRound: { zh: "第", en: "round", pt: "rodada" },
  drQueries: { zh: "次检索", en: "searches", pt: "buscas" },
  drSources: { zh: "个来源", en: "sources", pt: "fontes" },
  drExportPdf: { zh: "导出 PDF", en: "Export PDF", pt: "Exportar para PDF" },
  drExportMd: { zh: "导出 Markdown", en: "Export Markdown", pt: "Exportar para Markdown" },
  drBackToChat: { zh: "返回对话", en: "Back to chat", pt: "Voltar ao chat" },
  drPhasePlanning: { zh: "正在规划检索方向…", en: "Planning the research…", pt: "Planejando a pesquisa…" },
  drPhaseSearching: { zh: "正在联网检索…", en: "Searching the web…", pt: "Pesquisando na web…" },
  drPhaseReasoning: { zh: "正在分析并寻找信息缺口…", en: "Analyzing and finding gaps…", pt: "Analisando e encontrando lacunas…" },
  drPhaseWriting: { zh: "正在撰写报告…", en: "Writing the report…", pt: "Escrevendo o relatório…" },
  drPhaseDone: { zh: "完成", en: "Done", pt: "Concluído" },
  // first-launch setup
  setupBtn: { zh: "一键配置", en: "Set up for me", pt: "Configurar para mim" },
  setupTitle: { zh: "为这台电脑挑选模型", en: "Models picked for this machine", pt: "Modelos escolhidos para esta máquina" },
  setupBudget: { zh: "可用模型内存约", en: "model memory budget ≈", pt: "orçamento de memória do modelo ≈" },
  setupDownload: { zh: "下载", en: "Download", pt: "Baixar" },
  setupResolving: { zh: "正在查找…", en: "Resolving…", pt: "Localizando…" },
  setupLoad: { zh: "加载此模型", en: "Load this model", pt: "Carregar este modelo" },
  setupNotFound: { zh: "未找到合适的 GGUF 文件", en: "No suitable GGUF file found", pt: "Nenhum arquivo GGUF adequado encontrado" },
  setupStoreLink: { zh: "想用社区的其他模型?去模型商店逛逛", en: "Want other community models? Browse the model store", pt: "Procurando modelos da comunidade? Explore a loja de modelos" },
  setupFoot: {
    zh: "推荐按本机内存自动匹配规模与量化，下载自 HuggingFace，存入应用的 models 文件夹。",
    en: "Sized & quantized for your memory, downloaded from HuggingFace into the app's models folder.", pt: "Dimensionado e quantizado para a sua memória, baixado do HuggingFace diretamente na pasta de modelos."
  },
  // parameter tooltips
  tipTemperature: {
    zh: "随机性：越低回答越确定，越高越发散有创意（常用 0.7）",
    en: "Randomness: lower = more deterministic, higher = more creative (0.7 is typical)", pt: "Aleatoriedade: mais baixo = mais determinístico, mais alto = mais criativo (0.7 é típico)"
  },
  tipTopP: {
    zh: "核采样：只从累计概率前 P 的候选词中选择，与温度配合控制多样性",
    en: "Nucleus sampling: choose only from tokens within the top-P cumulative probability", pt: "Amostragem nucleus: escolhe apenas os tokens dentro da probabilidade cumulativa top-P"
  },
  tipTopK: {
    zh: "只考虑概率最高的 K 个候选词；0 = 关闭此限制",
    en: "Consider only the K most likely tokens; 0 disables the limit", pt: "Considera apenas os K tokens mais prováveis; 0 desativa o limite"
  },
  tipMinP: {
    zh: "过滤相对概率低于最高候选 P 倍的词；0 = 关闭",
    en: "Drop tokens whose probability is below P× the top token's; 0 disables", pt: "Descarta tokens com probabilidade inferior a P× a do token no topo; 0 desativa"
  },
  tipRepeatPenalty: {
    zh: "大于 1 时抑制重复用词；调得过高会伤害流畅度",
    en: ">1 discourages repetition; too high hurts fluency", pt: ">1 desencoraja repetições; muito alto prejudica a fluidez"
  },
  tipMaxTokens: {
    zh: "单次回复的最大 token 数；“不限制”时由上下文窗口决定",
    en: "Max tokens per reply; with no limit, the context window is the bound", pt: "Máximo de tokens por resposta; sem limite, a janela de contexto é a fronteira"
  },
  tipStopSeqs: {
    zh: "模型一旦输出这些字符串就立即停止生成",
    en: "Generation stops immediately when the model emits any of these strings", pt: "A geração para imediatamente caso o modelo emita qualquer uma dessas strings"
  },
  tipGpuAccel: {
    zh: "放到 GPU 的网络层数；自动 = 按显存放尽量多的层",
    en: "How many layers run on the GPU; Auto fills as many as memory allows", pt: "Quantas camadas rodam na GPU; Auto preenche o quanto a memória permitir"
  },
  tipCtxLength: {
    zh: "模型能“记住”的对话长度（token）；越大越耗内存",
    en: "How much conversation the model can hold (tokens); larger uses more memory", pt: "Quanto da conversa o modelo pode reter (tokens); valores maiores usam mais memória"
  },
  ejectingModel: { zh: "正在卸载旧模型…", en: "Ejecting old model…", pt: "Descarregando o modelo antigo…" },
  noLimit: { zh: "不限制", en: "No limit", pt: "Sem limite" },
  openModelsDir: { zh: "打开模型文件夹", en: "Open models folder", pt: "Abrir pasta de modelos" },
  voiceModelsDir: { zh: "语音模型文件夹", en: "Voice models folder" },
  voiceModelsDirHint: {
    zh: "离线语音模型存放在这里。自动下载失败时，可以按报错里的说明把文件手动放进来。",
    en: "Where the offline voice models are kept. If a download fails, you can put the files here by hand, as the error explains.",
  },
  voiceModelsDirOpen: { zh: "打开", en: "Open" },
  voiceDlFailed: { zh: "语音模型下载失败（{reason}）。", en: "Couldn't download the voice model ({reason})." },
  voiceDlMirror: {
    zh: "可以到「设置 → 模型 → HuggingFace 端点」换一个源（中国大陆推荐 hf-mirror.com）后再试。",
    en: "You can switch Settings → Model → HuggingFace endpoint to another source (hf-mirror.com in mainland China) and try again.",
  },
  voiceDlManual: {
    zh: "也可以手动下载：打开 {url}，下载 {files}，放进文件夹 {dir}，然后再试一次。",
    en: "Or fetch it by hand: from {url} download {files} into the folder {dir}, then try again.",
  },
  voiceDlAllFiles: { zh: "页面里的全部文件（dict 子文件夹保持原样）", en: "every file on that page (keep the dict subfolder)" },
  voiceDlArchive: {
    zh: "可以手动下载 {url}，解压到文件夹 {dir}（解压出的文件夹不要改名），然后再试一次。",
    en: "You can download {url} by hand and unpack it into the folder {dir} (keep the unpacked folder's name), then try again.",
  },
  // Model-facing: names the question after a turn's retrieved passages.
  questionLabel: { zh: "用户的问题：", en: "The user's question: ", pt: "Pergunta do usuário: " },
  webNoResults: {
    zh: "联网搜索没有找到与问题相关的网页，这次回答没有用到网络资料。",
    en: "Web search found nothing relevant to this question, so the answer doesn't use web sources.",
  },
  changeModelsDir: { zh: "更改位置", en: "Change location", pt: "Alterar local" },
  resetModelsDir: { zh: "恢复默认", en: "Reset to default", pt: "Restaurar padrão" },
  modelsDirHidden: {
    zh: "旧位置的 {n} 个模型不再显示在列表里（文件没有被移动或删除，点「恢复默认」即可重新看到）",
    en: "{n} model(s) in the previous locations are no longer listed. Nothing was moved or deleted — Reset to default lists them again.",
    pt: "{n} modelo(s) nos locais anteriores deixaram de ser listados. Nada foi movido ou excluído — Restaurar padrão os lista novamente.",
  },
  modelsDirMissing: {
    zh: "当前找不到这个文件夹（磁盘未连接？），暂时使用默认位置",
    en: "not reachable right now (disk not connected?) — using the default folder meanwhile",
    pt: "inacessível no momento (disco desconectado?) — usando a pasta padrão enquanto isso",
  },
  // stop reasons (shown in the stats line after generation)
  stopEos: { zh: "自然结束", en: "finished", pt: "concluído" },
  stopLength: { zh: "达到长度上限", en: "length limit", pt: "limite de comprimento" },
  stopContext: { zh: "上下文已满", en: "context full", pt: "contexto cheio" },
  stopStop: { zh: "命中停止词", en: "stop sequence", pt: "sequência de parada" },
  stopCancelled: { zh: "已手动停止", en: "cancelled", pt: "cancelado" },
  // hardware panel
  hwTitleBtn: { zh: "硬件信息", en: "Hardware", pt: "Hardware" },
  hwTitle: { zh: "本机硬件", en: "Hardware", pt: "Hardware" },
  hwCpu: { zh: "处理器", en: "CPU", pt: "CPU" },
  hwRam: { zh: "内存", en: "Memory", pt: "Memória" },
  hwGpu: { zh: "显卡", en: "GPU", pt: "GPU" },
  hwVram: { zh: "显存占用", en: "VRAM usage", pt: "Uso de VRAM" },
  hwBackend: { zh: "GPU 后端", en: "GPU backend", pt: "Backend da GPU" },
  hwAccel: { zh: "当前模型加速", en: "Current model", pt: "Modelo atual" },
  hwNoGpu: { zh: "未检测到独立显卡", en: "No discrete GPU detected", pt: "Nenhuma GPU dedicada detectada" },
  hwThreads: { zh: "{n} 线程", en: "{n} threads", pt: "{n} threads" },
  hwLayersOn: { zh: "{a}/{b} 层在 GPU", en: "{a}/{b} layers on GPU", pt: "{a}/{b} camadas na GPU" },
  hwCpuOnly: { zh: "纯 CPU 运行", en: "Running on CPU", pt: "Rodando na CPU" },
  hwNoModel: { zh: "未加载模型", en: "No model loaded", pt: "Nenhum modelo carregado" },
  // model info panel
  miTitleBtn: { zh: "模型信息", en: "Model info", pt: "Info do modelo" },
  miTitle: { zh: "模型信息", en: "Model info", pt: "Informações do modelo" },
  miName: { zh: "名称", en: "Name", pt: "Nome" },
  miArch: { zh: "架构", en: "Architecture", pt: "Arquitetura" },
  miParams: { zh: "参数量", en: "Parameters", pt: "Parâmetros" },
  miQuant: { zh: "量化", en: "Quantization", pt: "Quantização" },
  miSize: { zh: "大小", en: "Size", pt: "Tamanho" },
  miContext: { zh: "上下文", en: "Context", pt: "Contexto" },
  miTrained: { zh: "训练", en: "trained", pt: "treinado" },
  miLayers: { zh: "层数", en: "Layers", pt: "Camadas" },
  miEmbed: { zh: "嵌入维度", en: "Embedding", pt: "Embedding" },
  miEngine: { zh: "推理引擎", en: "Engine", pt: "Motor" },
  miTemplate: { zh: "对话模板", en: "Chat template", pt: "Template de chat" },
  miThinking: { zh: "思考推理", en: "Thinking", pt: "Raciocínio" },
  miTools: { zh: "工具调用", en: "Tool calls", pt: "Chamada de ferramentas" },
  miMultimodal: { zh: "多模态", en: "Multimodal", pt: "Multimodal" },
  miNoModel: { zh: "未加载模型", en: "No model loaded", pt: "Nenhum modelo carregado" },
  // load notices / OOM
  oomPartial: {
    zh: "显存不足，已自动减少 GPU 层数（{a}/{b} 层在 GPU）",
    en: "Low VRAM — GPU offload reduced to {a}/{b} layers", pt: "Pouca VRAM — aceleração por GPU reduzida para {a}/{b} camadas"
  },
  oomCpu: { zh: "显存不足，已回退到 CPU 运行", en: "Low VRAM — fell back to CPU", pt: "Pouca VRAM — caindo para a CPU" },
  cmRunInterrupted: {
    zh: "**上一次运行被中断** — 界面在运行过程中重新加载了(渲染进程崩溃或被重启),这一轮没有跑完。已完成的文件改动仍在工作区里,可以直接说「继续」。",
    en: "**The last run was interrupted** — the interface reloaded while it was working (a renderer crash or restart), so that turn never finished. Edits already written are still in the workspace; say \"continue\" to pick it up.", pt: "**A última execução foi interrompida** — a interface recarregou enquanto processava (travamento ou reinicialização), de forma que o turno nunca terminou. As edições já gravadas continuam no workspace; diga \"continuar\" para retomar."
  },
  ragTopK: { zh: "知识库引用条数", en: "Knowledge citations", pt: "Citações da base de conhecimento" },
  tipRagTopK: {
    zh: "一次提问最多引用多少条知识库片段。片段越多覆盖越全,但占用的上下文也越多。",
    en: "How many knowledge-base excerpts one question may cite. More covers more of the library and costs more context.", pt: "Quantos excertos da base de conhecimento uma pergunta pode citar. Mais alto cobre maior área da biblioteca, consumindo mais contexto."
  },
  kbCaptionImages: { zh: "识别文档内嵌图片", en: "Describe images inside documents", pt: "Descrever imagens em documentos" },
  kbCaptionImagesHint: {
    zh: "导入 PDF/Word/PPT 时，把文档里的图表交给视觉模型描述一遍，图表才能被检索到。会在已加载模型之外额外占用显存/内存——导入时闪退可以先关掉这项。",
    en: "When importing PDF/Word/PowerPoint, have the vision model describe the figures inside so charts are findable too. It runs on top of the model already loaded — turn it off first if importing crashes the app.",
    pt: "Ao importar PDF/Word/PowerPoint, o modelo de visão descreve as figuras internas para que gráficos também sejam encontrados. Consome memória além do modelo já carregado — desative primeiro se a importação travar o app.",
  },
  ragTopKHint: {
    zh: "文档多、问题跨文件时调高;上下文窗口小的模型建议调低。",
    en: "Raise it for a large library or questions that span files; lower it on a small context window.", pt: "Aumente para bibliotecas grandes ou perguntas que perpassam vários arquivos; diminua em janelas de contexto restritas."
  },
  errorLog: { zh: "错误日志", en: "Error log", pt: "Log de erro" },
  errorLogHint: {
    zh: "应用异常会自动记录到 chaty-error.log。提 issue 时请附上这个文件，能大幅加快定位。",
    en: "App errors are recorded to chaty-error.log automatically. Please attach it when filing an issue — it speeds up diagnosis a lot.", pt: "Os erros da aplicação são registrados no chaty-error.log automaticamente. Anexe-o quando for abrir uma issue — acelera bastante o diagnóstico."
  },
  errorLogOpen: { zh: "打开错误日志", en: "Open error log", pt: "Abrir log de erro" },
  errorLogClear: { zh: "清空日志", en: "Clear log", pt: "Limpar log" },
  confirmClearErrorLog: {
    zh: "清空错误日志?已记录的报错会全部删除,之后新的报错仍会照常记录。",
    en: "Clear the error log? Everything recorded so far is deleted; new errors are still recorded afterwards.",
    pt: "Limpar o log de erro? Tudo o que foi registrado até agora será apagado; novos erros continuarão sendo registrados."
  },
  conversionSuspect: {
    zh: "该模型家族与当前内置引擎的兼容性存在已知问题(转换元数据或引擎版本尚未跟上),回复可能退化为空白或乱码。macOS 上请改用该模型的 MLX 版本(运行正常);GGUF 需等待后续版本升级引擎。",
    en: "This model family has known compatibility issues with the bundled engine (conversion metadata or engine version lag) — replies may degenerate into blanks or noise. On macOS use the model's MLX build (works well); GGUF support awaits an engine upgrade.", pt: "Esta família de modelos tem problemas de compatibilidade conhecidos com o motor (falha de conversão ou atraso na versão) — as respostas podem degenerar em ruído ou vazio. No macOS, prefira o modelo MLX (funciona bem); o suporte GGUF requer atualização do motor."
  },
  visionConfigMissing: {
    zh: "该模型目录缺少图像处理器配置(preprocessor_config.json),视觉功能已停用,本次以纯文本模式加载。从官方模型仓库补齐该文件后重新加载即可恢复视觉能力。",
    en: "This model folder is missing its image-processor configuration (preprocessor_config.json), so vision is disabled and the model was loaded text-only. Restore that file from the official model repo and reload to re-enable vision.", pt: "Falta a configuração do processador de imagem (preprocessor_config.json) nesta pasta de modelo, então a visão está desativada e o modelo foi carregado apenas com texto. Restaure esse arquivo do repositório oficial e recarregue para reativar a visão."
  },
  gpuCrashCpu: {
    zh: "上次加载模型时 GPU 驱动导致程序崩溃，本次已改用 CPU 运行（速度较慢但稳定）。更新显卡驱动后可尝试恢复 GPU。",
    en: "The GPU driver crashed the app during the last model load — running on CPU this time (slower but stable). Update your GPU driver to try GPU again.", pt: "O driver da GPU travou a aplicação durante o carregamento do modelo — caindo para CPU desta vez (mais lento, mas estável). Atualize o driver da placa de vídeo para tentar GPU novamente."
  },
  gpuCrashCapped: {
    zh: "上次加载模型时 GPU 驱动导致程序崩溃，本次已降低 GPU 层数({a}/{b} 层)以求稳妥,速度会慢一些。下一次成功加载后会自动恢复满配;若反复崩溃请更新显卡驱动。",
    en: "The GPU driver crashed the app during a previous model load, so this one runs with fewer layers on the GPU ({a}/{b}) — slower, but stable. The next load that survives restores full offload; if it keeps crashing, update your GPU driver.", pt: "O driver da GPU travou o aplicativo em um carregamento anterior, então este modelo roda com menos camadas na GPU ({a}/{b}) — mais lento, porém estável. O próximo carregamento bem-sucedido restaurará a aceleração total; se continuar travando, atualize seu driver de GPU."
  },
  ctxClamped: {
    zh: "上下文已按内存自动调整为 {n}（模型权重 + KV 缓存需放入统一内存）",
    en: "Context auto-fitted to {n} (weights + KV cache must fit in unified memory)", pt: "Contexto ajustado automaticamente para {n} (pesos + cache KV devem caber na memória unificada)"
  },
  oomFail: {
    zh: "内存不足，无法加载该模型。试试更小 / 更高量化的模型，或关闭其他占用内存的程序。",
    en: "Out of memory — couldn't load this model. Try a smaller / more-quantized model, or free up RAM.", pt: "Falta de memória — não foi possível carregar o modelo. Tente um modelo menor ou mais quantizado, ou libere RAM."
  },
  toastDismiss: { zh: "点击关闭", en: "Click to dismiss", pt: "Clique para fechar" },
  // greetings (time of day)
  greetMorning: { zh: "早上好", en: "Good morning", pt: "Bom dia" },
  greetNoon: { zh: "中午好", en: "Good afternoon", pt: "Boa tarde" },
  greetAfternoon: { zh: "下午好", en: "Good afternoon", pt: "Boa tarde" },
  greetEvening: { zh: "晚上好", en: "Good evening", pt: "Boa noite" },
  greetNight: { zh: "夜深了", en: "Working late", pt: "Trabalhando até tarde" },
  // model-facing prompts
  todayNote: {
    zh: "当前日期是 {date}。涉及“今天/最近/现在”等时间时以此为准。",
    en: 'Today is {date}. When the question refers to "today/recent/now", use this date.',
  },
  webInstruction: {
    zh: "下面是联网检索到的 {n} 条资料，按【1】到【{n}】编号。请综合它们，用自然连贯的语言直接回答用户的问题；在用到某条资料的句子末尾标注对应角标，如【1】或【1】【3】（不要写“来源”二字，只写数字角标；编号最大是【{n}】，没有更多资料）。若资料不足以回答，请直说。\n\n",
    en: "Below are {n} pieces of information retrieved from the web, numbered 【1】 to 【{n}】. Use them to answer the user's question in natural prose, and append the matching citation marker(s) — e.g. 【1】 or 【1】【3】 — at the end of each sentence that draws on a source; there is nothing beyond 【{n}】. If the material is insufficient, say so.\n\n", pt: "Abaixo estão {n} informações extraídas da web, enumeradas de 【1】 a 【{n}】. Utilize-as para responder à pergunta do usuário num texto natural e acrescente os marcadores de citação equivalentes — ex: 【1】 ou 【1】【3】 — no final de cada frase que se basear em uma fonte; não há nada além de 【{n}】. Se o material for insuficiente, diga explicitamente.\n\n"
  },
  ragInstruction: {
    zh: "下面是检索到的 {n} 个文档片段，按【1】到【{n}】编号（编号最大是【{n}】）。严格依据这些片段回答：只陈述片段中明确支持的内容，绝不编造、不引入片段之外的事实或数字；若片段不足以回答，必须直接说明“当前文档未提及”。在用到某条片段的句子末尾标注对应角标，如【1】或【1】【3】（只写数字角标）。\n\n",
    en: "Below are {n} retrieved passages, numbered 【1】 to 【{n}】 (there is nothing beyond 【{n}】). Answer STRICTLY from these passages: state only what they explicitly support, never invent facts or numbers beyond them; if they do not contain the answer, you must say the current documents do not mention it. Append the matching citation marker(s) — e.g. 【1】 or 【1】【3】 — at the end of each sentence that draws on a passage.\n\n", pt: "Abaixo estão trechos extraídos da base de conhecimento local do usuário, enumerados como 【1】【2】…. Responda ESTRITAMENTE a partir destes trechos: declare apenas o que eles suportam explicitamente, nunca invente fatos ou números além disso; se não contiverem a resposta, você deve afirmar que os documentos atuais não a mencionam. Acrescente os marcadores de citação equivalentes — ex: 【1】 ou 【1】【3】 — no final de cada frase baseada num trecho.\n\n"
  },
  attachInstruction: {
    zh: "用户上传了文件《{name}》，其内容如下，回答时请优先依据它：\n\n",
    en: 'The user uploaded a file "{name}". Its content is below; base your answer on it:\n\n',
  },
  // ---- Text-to-image (image studio) ----
  modeImage: { zh: "文生图", en: "Image", pt: "Imagem" },
  imgModeTip: { zh: "已加载文生图模型，整个应用处于生图模式。加载对话模型即可回到对话与编程。", en: "An image model is loaded, so the app is in image mode. Load a chat model to get Chat and Code back.", pt: "Um modelo de imagem está carregado, então o app está no modo imagem. Carregue um modelo de chat para voltar ao Chat e ao Código." },
  imageBadge: { zh: "文生图", en: "Image", pt: "Imagem" },
  imageBadgeTip: { zh: "文生图模型——加载后进入生图模式", en: "Text-to-image model — loading it opens the image studio", pt: "Modelo de texto para imagem — carregá-lo abre o estúdio de imagens" },
  imgMissingBadgeTip: { zh: "还缺少 VAE 或文本编码器等配套文件，加载时会提示下载", en: "Still missing a companion file (VAE, text encoder) — you'll be offered the download when loading", pt: "Ainda falta um arquivo complementar (VAE, codificador de texto) — o download será oferecido ao carregar" },
  imgNew: { zh: "新建生图", en: "New session", pt: "Nova sessão" },
  imgSearch: { zh: "搜索生图会话…", en: "Search sessions…", pt: "Pesquisar sessões…" },
  imgNoHistory: { zh: "暂无生图会话", en: "No image sessions yet", pt: "Nenhuma sessão de imagem ainda" },
  imgCmdkHint: { zh: "生图会话", en: "Image session", pt: "Sessão de imagem" },
  imgDeleteSession: { zh: "删除会话", en: "Delete session", pt: "Excluir sessão" },
  imgDeleteSessionConfirm: {
    zh: "将删除这个会话的每一轮和它们生成的图片文件，无法撤销。",
    en: "Every round in this session and the picture files it made will be deleted. This can't be undone.",
    pt: "Todas as rodadas desta sessão e os arquivos de imagem gerados serão excluídos. Não pode ser desfeito.",
  },
  imgAccel: { zh: "采样加速", en: "Sampling speed-up", pt: "Aceleração da amostragem" },
  imgAccelHint: {
    zh: "复用相邻步骤里几乎没变的计算，出图更快，细节略有损失；步数很少的 Turbo 类模型收益有限",
    en: "Reuses work between steps that barely change: faster pictures, slightly less detail. Little gain on few-step Turbo models",
    pt: "Reaproveita o trabalho entre passos que quase não mudam: imagens mais rápidas, um pouco menos de detalhe. Pouco ganho em modelos Turbo de poucos passos",
  },
  imgAccelBalanced: { zh: "均衡", en: "Balanced", pt: "Equilibrado" },
  imgAccelFast: { zh: "更快", en: "Faster", pt: "Mais rápido" },
  imgAccelChip: { zh: "加速 · {level}", en: "Speed-up · {level}", pt: "Aceleração · {level}" },
  imgCacheEncode: { zh: "编码已复用缓存", en: "prompt encoding cached", pt: "codificação em cache" },
  imgCacheSteps: { zh: "缓存跳过 {n}/{total} 步", en: "{n}/{total} steps cached", pt: "{n}/{total} passos em cache" },
  imgAutoChain: { zh: "多轮改图", en: "Multi-turn editing", pt: "Edição em várias rodadas" },
  imgAutoChainHint: {
    zh: "支持改图的模型：每轮画完后，下一条提示词默认接着改这一轮的图",
    en: "Editing models: after each round, the next prompt edits its picture",
    pt: "Modelos de edição: após cada rodada, o próximo prompt edita a imagem dela",
  },
  imgRefFrom: { zh: "改自第 {n} 轮", en: "From round {n}", pt: "Da rodada {n}" },
  imgJumpToRound: { zh: "跳到那一轮", en: "Go to that round", pt: "Ir para essa rodada" },
  imgHero: { zh: "开始创作", en: "Start creating", pt: "Comece a criar" },
  imgHeroSub: { zh: "{model} · 默认 {size}，描述你想要的画面", en: "{model} · {size} by default — describe the picture you want", pt: "{model} · {size} por padrão — descreva a imagem que você quer" },
  imgPromptPh: { zh: "描述画面，Enter 生成…", en: "Describe the picture — Enter to generate…", pt: "Descreva a imagem — Enter para gerar…" },
  imgPromptPhMod: { zh: "描述画面，⌘/Ctrl+Enter 生成…", en: "Describe the picture — ⌘/Ctrl+Enter to generate…", pt: "Descreva a imagem — ⌘/Ctrl+Enter para gerar…" },
  imgGenerate: { zh: "生成", en: "Generate", pt: "Gerar" },
  imgStop: { zh: "停止", en: "Stop", pt: "Parar" },
  imgStopAfter: { zh: "画完这张后停止", en: "Stop after this one", pt: "Parar após esta" },
  imgStopping: { zh: "正在停止…", en: "Stopping…", pt: "Parando…" },
  imgStopped: { zh: "已停止生成", en: "Generation stopped", pt: "Geração interrompida" },
  imgStageEncode: { zh: "理解提示词", en: "Reading the prompt", pt: "Lendo o prompt" },
  imgStageWeights: { zh: "加载权重", en: "Loading weights", pt: "Carregando pesos" },
  imgStageSample: { zh: "绘制中", en: "Drawing", pt: "Desenhando" },
  imgStageDecode: { zh: "解码图像", en: "Decoding the image", pt: "Decodificando a imagem" },
  imgPicN: { zh: "第 {i}/{n} 张", en: "Picture {i}/{n}", pt: "Imagem {i}/{n}" },
  imgStepsN: { zh: "{n} 步", en: "{n} steps", pt: "{n} passos" },
  imgCountN: { zh: "{n} 张", en: "{n} pics", pt: "{n} imgs" },
  imgSecPerStep: { zh: "秒/步", en: "s/step", pt: "s/passo" },
  imgStepPerSec: { zh: "步/秒", en: "steps/s", pt: "passos/s" },
  imgElapsed: { zh: "已用 {t}", en: "{t} elapsed", pt: "{t} decorridos" },
  imgEta: { zh: "剩余约 {t}", en: "~{t} left", pt: "~{t} restantes" },
  imgSeed: { zh: "种子", en: "Seed", pt: "Semente" },
  imgSeedRandom: { zh: "随机种子", en: "Random seed", pt: "Semente aleatória" },
  imgSeedFixed: { zh: "固定", en: "Fixed", pt: "Fixa" },
  imgSeedLocked: { zh: "固定种子：同样的提示词和参数会画出同样的图。点击改回随机", en: "Fixed seed: the same prompt and settings draw the same picture. Click for random", pt: "Semente fixa: o mesmo prompt e configurações geram a mesma imagem. Clique para aleatória" },
  imgSeedDice: { zh: "换一个随机种子", en: "Roll a new seed", pt: "Sortear nova semente" },
  imgSeedHint: { zh: "随机：每次都不同；固定：可复现同一张图", en: "Random: different every time. Fixed: reproduces the same picture", pt: "Aleatória: diferente a cada vez. Fixa: reproduz a mesma imagem" },
  imgActZoom: { zh: "查看大图", en: "View full size", pt: "Ver em tamanho real" },
  imgActSave: { zh: "另存为…", en: "Save as…", pt: "Salvar como…" },
  imgActCopy: { zh: "复制图片", en: "Copy image", pt: "Copiar imagem" },
  imgCopied: { zh: "已复制", en: "Copied", pt: "Copiado" },
  imgActReveal: { zh: "在文件夹中显示", en: "Show in folder", pt: "Mostrar na pasta" },
  imgActUseRef: { zh: "用作参考图", en: "Use as reference", pt: "Usar como referência" },
  imgActEdit: { zh: "继续改这张", en: "Edit this further", pt: "Continuar editando" },
  imgActReuseSeed: { zh: "固定这个种子", en: "Keep this seed", pt: "Fixar esta semente" },
  imgAgain: { zh: "再来一次", en: "Again", pt: "De novo" },
  imgAgainTitle: { zh: "同样的提示词和参数，换个种子再画一轮", en: "Same prompt and settings, a new seed, as a new round", pt: "Mesmo prompt e configurações, nova semente, em uma nova rodada" },
  imgReuse: { zh: "复用提示词与参数", en: "Reuse prompt & settings", pt: "Reusar prompt e configurações" },
  imgDelete: { zh: "删除", en: "Delete", pt: "Excluir" },
  imgDeleteTitle: { zh: "删除这一轮", en: "Delete this round", pt: "Excluir esta rodada" },
  imgDeleteConfirm: { zh: "将删除这一轮和它生成的图片文件，无法撤销。", en: "This removes the round and the picture files it made. It can't be undone.", pt: "Isso remove a rodada e os arquivos de imagem gerados. Não pode ser desfeito." },
  imgImages: { zh: "图片", en: "Images", pt: "Imagens" },
  imgRefAdd: { zh: "添加参考图", en: "Add a reference picture", pt: "Adicionar imagem de referência" },
  imgRefRemove: { zh: "移除参考图", en: "Remove the reference", pt: "Remover a referência" },
  imgRefEdit: { zh: "参考图（按提示词编辑这张图）", en: "Reference (edited per the prompt)", pt: "Referência (editada conforme o prompt)" },
  imgRefInit: { zh: "参考图（以此为起点重绘）", en: "Reference (redrawn from this start)", pt: "Referência (redesenhada a partir daqui)" },
  imgStrength: { zh: "重绘强度", en: "Redraw strength", pt: "Força do redesenho" },
  imgStrengthHint: { zh: "图生图时离参考图多远：越小越像原图，1 = 几乎完全重画", en: "How far img2img moves from the reference: lower stays closer, 1 redraws almost entirely", pt: "Quanto o img2img se afasta da referência: menor fica mais próximo, 1 redesenha quase tudo" },
  imgDropRef: { zh: "松开以设为参考图", en: "Drop to use as the reference picture", pt: "Solte para usar como imagem de referência" },
  imgDropNotImage: { zh: "只能拖入图片作为参考图", en: "Only a picture can be dropped here as a reference", pt: "Apenas uma imagem pode ser usada como referência" },
  imgNegative: { zh: "负面提示词", en: "Negative prompt", pt: "Prompt negativo" },
  imgNegativePh: { zh: "不想出现的内容，如：模糊、变形、水印", en: "What to avoid, e.g. blurry, deformed, watermark", pt: "O que evitar, ex.: borrado, deformado, marca d'água" },
  imgNegativeDefault: { zh: "默认负面提示词", en: "Default negative prompt", pt: "Prompt negativo padrão" },
  imgNegativeTip: { zh: "每次生成都会带上的负面提示词，可在生图界面临时修改", en: "Sent with every generation; can be changed per generation in the studio", pt: "Enviado em cada geração; pode ser alterado por geração no estúdio" },
  imgNegativeUnused: { zh: "当前模型以 CFG 1 运行，负面提示词不起作用", en: "The loaded model runs at CFG 1, where a negative prompt has no effect", pt: "O modelo carregado roda com CFG 1, onde o prompt negativo não tem efeito" },
  imgAspect: { zh: "画面比例", en: "Aspect ratio", pt: "Proporção" },
  imgAspectCustom: { zh: "自定义", en: "Custom", pt: "Personalizado" },
  imgCustomSize: { zh: "自定义尺寸（宽 × 高）", en: "Custom size (width × height)", pt: "Tamanho personalizado (largura × altura)" },
  imgAlignHint: { zh: "会对齐到 {n} 的倍数", en: "Snapped to multiples of {n}", pt: "Ajustado a múltiplos de {n}" },
  imgResolution: { zh: "分辨率", en: "Resolution", pt: "Resolução" },
  imgResolutionTip: { zh: "按正方形边长计的像素量，比例不同时面积保持一致。自动 = 模型训练时的分辨率", en: "Pixel budget as the side of a square; other ratios keep the same area. Auto = what the model was trained at", pt: "Orçamento de pixels como o lado de um quadrado; outras proporções mantêm a área. Auto = a resolução de treino do modelo" },
  imgAuto: { zh: "自动", en: "Auto", pt: "Auto" },
  imgBatch: { zh: "每次张数", en: "Pictures per run", pt: "Imagens por execução" },
  imgSteps: { zh: "步数", en: "Steps", pt: "Passos" },
  imgStepsTip: { zh: "去噪迭代次数：越多越精细也越慢。蒸馏/Turbo 模型只需很少几步", en: "Denoising iterations: more is finer and slower. Distilled/Turbo models need only a few", pt: "Iterações de remoção de ruído: mais é mais fino e mais lento. Modelos destilados/Turbo precisam de poucas" },
  imgCfgTip: { zh: "提示词引导强度：越高越贴近提示词，过高会过饱和。Turbo/FLUX 类模型用 1", en: "How strongly the prompt steers: higher follows it closer, too high oversaturates. Turbo/FLUX-style models use 1", pt: "Quanto o prompt guia: mais alto segue mais de perto, alto demais satura. Modelos Turbo/FLUX usam 1" },
  imgGuidance: { zh: "蒸馏引导", en: "Distilled guidance", pt: "Guidance destilada" },
  imgGuidanceTip: { zh: "FLUX-dev 这类蒸馏模型的引导强度（它们的 CFG 固定为 1）", en: "Guidance for distilled models such as FLUX-dev (whose CFG stays at 1)", pt: "Guidance para modelos destilados como FLUX-dev (cujo CFG fica em 1)" },
  imgFlowShift: { zh: "Flow shift", en: "Flow shift", pt: "Flow shift" },
  imgFlowShiftTip: { zh: "Flow 类模型（Qwen-Image、FLUX、SD3、Z-Image）的噪声调度偏移，一般保持推荐值", en: "Noise-schedule shift for flow models (Qwen-Image, FLUX, SD3, Z-Image); usually best left recommended", pt: "Deslocamento do agendamento de ruído para modelos flow (Qwen-Image, FLUX, SD3, Z-Image); geralmente melhor no recomendado" },
  imgSampler: { zh: "采样器", en: "Sampler", pt: "Amostrador" },
  imgSamplerHint: { zh: "自动 = 模型推荐的采样器", en: "Auto = the model's recommended sampler", pt: "Auto = o amostrador recomendado do modelo" },
  imgScheduler: { zh: "调度器", en: "Scheduler", pt: "Agendador" },
  imgSchedulerHint: { zh: "噪声调度曲线，自动 = 模型推荐", en: "Noise schedule; Auto = the model's recommendation", pt: "Agendamento de ruído; Auto = recomendação do modelo" },
  imgRecommended: { zh: "推荐", en: "Recommended", pt: "Recomendado" },
  imgResetRecommended: { zh: "恢复推荐值", en: "Back to recommended", pt: "Voltar ao recomendado" },
  imgMoreSettings: { zh: "更多设置…", en: "More settings…", pt: "Mais configurações…" },
  imgClipSkipTip: { zh: "跳过 CLIP 最后几层，部分 SD 1.x/XL 微调模型推荐 2", en: "Skip CLIP's last layers; some SD 1.x/XL fine-tunes recommend 2", pt: "Pular as últimas camadas do CLIP; alguns ajustes finos SD 1.x/XL recomendam 2" },
  imgPreview: { zh: "实时预览", en: "Live preview", pt: "Pré-visualização ao vivo" },
  imgPreviewHint: { zh: "绘制时显示画面雏形。精确预览每步都解码一次，更慢", en: "Shows the picture forming while it draws. Exact decodes every step and is slower", pt: "Mostra a imagem se formando. Exata decodifica a cada passo e é mais lenta" },
  imgPreviewFast: { zh: "快速", en: "Fast", pt: "Rápida" },
  imgPreviewExact: { zh: "精确", en: "Exact", pt: "Exata" },
  imgPreviewEvery: { zh: "每隔几步预览一次", en: "Preview every N steps", pt: "Pré-visualizar a cada N passos" },
  imgVaeTiling: { zh: "VAE 分块解码", en: "Tiled VAE decoding", pt: "Decodificação VAE em blocos" },
  imgVaeTilingHint: { zh: "大尺寸出图时分块解码，显著省显存，稍慢", en: "Decode large pictures in tiles — much less VRAM, a little slower", pt: "Decodifica imagens grandes em blocos — muito menos VRAM, um pouco mais lento" },
  imgFormat: { zh: "保存格式", en: "File format", pt: "Formato do arquivo" },
  imgFormatHint: { zh: "PNG 无损并把提示词与参数写进图片；JPG 更小", en: "PNG is lossless and carries the prompt and settings inside; JPG is smaller", pt: "PNG é sem perdas e guarda o prompt e as configurações; JPG é menor" },
  imgOutputDir: { zh: "图片保存位置", en: "Where pictures are saved", pt: "Onde as imagens são salvas" },
  imgOutputDirHint: { zh: "默认按日期存放在数据文件夹的 images 目录", en: "By default, by date under images/ in the data folder", pt: "Por padrão, por data em images/ na pasta de dados" },
  imgOpenFolder: { zh: "打开文件夹", en: "Open folder", pt: "Abrir pasta" },
  imgSetNoModel: { zh: "加载文生图模型后，这里的「自动」会显示该模型的推荐值。", en: "Load an image model and each Auto here shows that model's recommendation.", pt: "Carregue um modelo de imagem e cada Auto aqui mostrará a recomendação dele." },
  setCatImageGen: { zh: "生图参数", en: "Image generation", pt: "Geração de imagem" },
  setCatImageModel: { zh: "生图模型", en: "Image model", pt: "Modelo de imagem" },
  imgManageComponents: { zh: "管理配套文件…", en: "Companion files…", pt: "Arquivos complementares…" },
  imgDevice: { zh: "计算设备", en: "Device", pt: "Dispositivo" },
  imgDeviceTip: { zh: "默认全部放到 GPU，放不下的部分才自动溢出到内存", en: "Everything on the GPU by default; only what doesn't fit spills to RAM", pt: "Tudo na GPU por padrão; só o que não cabe vai para a RAM" },
  imgDeviceGpu: { zh: "GPU 加速", en: "GPU", pt: "GPU" },
  imgDeviceCpu: { zh: "仅 CPU", en: "CPU only", pt: "Só CPU" },
  imgDeviceHint: { zh: "改动后点「重新加载以应用」生效。仅 CPU 可以运行，但一张图可能要几分钟。", en: "Reload to apply. CPU only works, but a picture may take minutes.", pt: "Recarregue para aplicar. Só CPU funciona, mas uma imagem pode levar minutos." },
  imgOffload: { zh: "省显存模式", en: "Save VRAM", pt: "Economizar VRAM" },
  imgOffloadHint: { zh: "权重常驻内存、用到时再搬到 GPU。显存不够时开启，速度会变慢", en: "Keep weights in RAM and move them to the GPU as needed. For small GPUs; slower", pt: "Mantém os pesos na RAM e os move para a GPU quando necessário. Para GPUs pequenas; mais lento" },
  imgTeCpu: { zh: "文本编码器放在 CPU", en: "Text encoder on the CPU", pt: "Codificador de texto na CPU" },
  imgTeCpuHint: { zh: "省下数 GB 显存，理解提示词会慢一些", en: "Saves several GB of VRAM; reading the prompt gets slower", pt: "Economiza vários GB de VRAM; ler o prompt fica mais lento" },
  imgVaeCpu: { zh: "VAE 放在 CPU", en: "VAE on the CPU", pt: "VAE na CPU" },
  imgVaeCpuHint: { zh: "省显存，解码图像会变慢", en: "Saves VRAM; decoding gets slower", pt: "Economiza VRAM; a decodificação fica mais lenta" },
  imgFlashAttn: { zh: "Flash Attention", en: "Flash Attention", pt: "Flash Attention" },
  imgFlashAttnHint: { zh: "去噪模型用 Flash Attention，更快更省显存", en: "Flash attention in the denoiser — faster, less VRAM", pt: "Flash attention no denoiser — mais rápido, menos VRAM" },
  imgMaxVram: { zh: "显存上限", en: "VRAM budget", pt: "Limite de VRAM" },
  imgMaxVramTip: { zh: "给生图引擎的显存预算，超出的部分放到内存。不限 = 用尽当前空闲显存", en: "VRAM the image engine may use; beyond it goes to RAM. No limit = all free VRAM", pt: "VRAM que o motor de imagem pode usar; o excedente vai para a RAM. Sem limite = toda a VRAM livre" },
  imgThreads: { zh: "CPU 线程数", en: "CPU threads", pt: "Threads da CPU" },
  imgMmap: { zh: "内存映射加载", en: "Memory-mapped loading", pt: "Carregamento mapeado em memória" },
  imgMmapHint: { zh: "直接映射模型文件，加载更快、少占一份内存", en: "Map the model file directly — faster loads, one less copy in RAM", pt: "Mapeia o arquivo diretamente — carrega mais rápido, uma cópia a menos na RAM" },
  imgGpuCrashCpu: { zh: "图像引擎在 GPU 上加载时崩溃（多为显卡驱动问题），已自动改用 CPU 运行，速度会慢很多。更新显卡驱动后可在设置里切回 GPU。", en: "The image engine crashed loading onto the GPU (usually a driver problem) and fell back to the CPU, which is much slower. After updating your GPU driver, switch back in Settings.", pt: "O motor de imagem travou ao carregar na GPU (geralmente um problema de driver) e passou para a CPU, bem mais lenta. Após atualizar o driver, volte à GPU nas configurações." },
  imgCompTitle: { zh: "配套文件", en: "Companion files", pt: "Arquivos complementares" },
  imgCompNeedHint: { zh: "文生图 GGUF 只包含去噪模型，还需要下面标记的文件才能出图。可以一键下载到模型文件夹，也可以手动指定已有文件。", en: "A text-to-image GGUF holds only the denoiser; it needs the files marked below to draw. Download them into the model's folder in one click, or pick files you already have.", pt: "Um GGUF de texto para imagem contém só o denoiser; precisa dos arquivos marcados abaixo. Baixe-os para a pasta do modelo com um clique, ou escolha arquivos que você já tem." },
  imgCompReady: { zh: "需要的文件都已就位。", en: "Everything it needs is in place.", pt: "Tudo o que ele precisa está no lugar." },
  imgCompNone: { zh: "这个模型不需要额外文件。", en: "This model needs no other files.", pt: "Este modelo não precisa de outros arquivos." },
  imgOptional: { zh: "可选", en: "optional", pt: "opcional" },
  imgWillDownload: { zh: "将下载", en: "Will download", pt: "Vai baixar" },
  imgPickManually: { zh: "未找到，请手动指定", en: "Not found — pick it manually", pt: "Não encontrado — escolha manualmente" },
  imgNotUsed: { zh: "未使用", en: "Not used", pt: "Não usado" },
  imgPickFile: { zh: "选择文件…", en: "Choose file…", pt: "Escolher arquivo…" },
  imgAutoDetect: { zh: "自动识别", en: "Auto-detect", pt: "Detectar" },
  imgDownloadMissing: { zh: "下载缺少的文件（{size}）", en: "Download what's missing ({size})", pt: "Baixar o que falta ({size})" },
  imgLoadModel: { zh: "加载模型", en: "Load model", pt: "Carregar modelo" },
  imgWeightsFiles: { zh: "模型文件", en: "Model files", pt: "Arquivos de modelo" },
  imgSrcOverride: { zh: "手动指定", en: "chosen", pt: "escolhido" },
  imgSrcShared: { zh: "来自其他模型文件夹", en: "from another model's folder", pt: "da pasta de outro modelo" },
  imgSrcFolder: { zh: "同一文件夹", en: "same folder", pt: "mesma pasta" },
  imgRoleVae: { zh: "VAE", en: "VAE", pt: "VAE" },
  imgRoleLlm: { zh: "文本编码器", en: "Text encoder", pt: "Codificador de texto" },
  imgRoleLlmVision: { zh: "视觉编码器（参考图编辑）", en: "Vision encoder (reference editing)", pt: "Codificador visual (edição por referência)" },
  imgRoleClipL: { zh: "CLIP-L 文本编码器", en: "CLIP-L text encoder", pt: "Codificador de texto CLIP-L" },
  imgRoleClipG: { zh: "CLIP-G 文本编码器", en: "CLIP-G text encoder", pt: "Codificador de texto CLIP-G" },
  imgRoleT5: { zh: "T5-XXL 文本编码器", en: "T5-XXL text encoder", pt: "Codificador de texto T5-XXL" },
  imgClearHistory: { zh: "清空生图会话", en: "Clear image sessions", pt: "Limpar sessões de imagem" },
  imgClearHistoryHint: { zh: "删除所有生图会话及其图片文件", en: "Deletes every image session and its picture files", pt: "Exclui todas as sessões de imagem e seus arquivos" },
  imgClearHistoryConfirm: { zh: "将删除全部生图会话和其中的图片文件，无法撤销。确定吗？", en: "Every image session and the picture files in it will be deleted. This can't be undone. Continue?", pt: "Todas as sessões de imagem e os arquivos nelas serão excluídos. Não pode ser desfeito. Continuar?" },
  statImages: { zh: "生成图片", en: "Images", pt: "Imagens" },
  miImageEngine: { zh: "识别为", en: "Identified as", pt: "Identificado como" },
  miImageDefaults: { zh: "推荐参数", en: "Recommended", pt: "Recomendado" },
  miImageEdit: { zh: "参考图编辑", en: "Reference editing", pt: "Edição por referência" },
  hwImageGpu: { zh: "GPU 加速（全部组件）", en: "GPU (all components)", pt: "GPU (todos os componentes)" },
  storeTask: { zh: "类型", en: "Type", pt: "Tipo" },
  storeTaskAll: { zh: "全部", en: "All", pt: "Todos" },
  storeTaskImage: { zh: "文生图", en: "Text to image", pt: "Texto para imagem" },
  storeImageIncluded: { zh: "含 VAE 与文本编码器", en: "VAE and text encoder included", pt: "VAE e codificador de texto incluídos" },
  storeImageManual: { zh: "配套文件需加载时再选", en: "companion files chosen at load", pt: "arquivos complementares escolhidos ao carregar" },
} satisfies Record<string, Entry>;

export type TKey = keyof typeof T;

export function detectLang(): Lang {
  try {
    const saved = localStorage.getItem(LANG_KEY);
    if (LANGS.some((l) => l.id === saved)) return saved as Lang;
  } catch {
    /* ignore */
  }
  // English remains the default; voice features support Chinese as well.
  return "en";
}

/// Pure resolution (testable without the provider). Community locales are
/// allowed to be partial — a missing translation shows English, never a
/// blank or a raw key.
export function lookup(key: TKey, lang: Lang, vars?: Record<string, string | number>): string {
  const e: Entry | undefined = T[key];
  let s: string = e?.[lang] ?? e?.en ?? key;
  if (vars) {
    for (const k of Object.keys(vars)) s = s.split(`{${k}}`).join(String(vars[k]));
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
