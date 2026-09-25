<div align="center">

[English](README.md) · [简体中文](README.zh-CN.md) · **Português (BR)**

<img src="icon.png" width="84" height="84" alt="Chaty" />

# Chaty

**Os modelos no seu disco, trabalhando de verdade.**

Chat, um agente de programação, um estúdio de imagens, respostas a partir dos seus próprios documentos e uma voz para conversar —<br />
com modelos abertos rodando inteiramente no seu Mac ou PC. Sem conta, sem nuvem, sem telemetria.

[![Release](https://img.shields.io/github/v/release/Fangyuan025/Chaty?label=release&color=3a3a3a)](../../releases/latest)
[![Downloads](https://img.shields.io/github/downloads/Fangyuan025/Chaty/total?color=3a3a3a&cacheSeconds=3600)](../../releases)
[![CI](https://img.shields.io/github/actions/workflow/status/Fangyuan025/Chaty/ci.yml?branch=main&label=CI)](../../actions)
[![License: MIT](https://img.shields.io/badge/license-MIT-3a3a3a)](LICENSE)

[**Baixar**](../../releases/latest) · [**Site**](https://chaty.ca) · [**Documentação**](https://chaty.ca/docs.html) · [**Changelog**](CHANGELOG.md)

<sub>macOS (Apple Silicon) · Windows 10/11 · Linux (AppImage) — GGUF no llama.cpp, MLX nativo no Apple Silicon, texto para imagem no stable-diffusion.cpp</sub>

<br />

<picture>
  <source media="(prefers-color-scheme: light)" srcset="docs/screenshots/shot-code-light.jpg" />
  <img src="docs/screenshots/shot-code.jpg" width="900" alt="Modo Code do Chaty: um plano de tarefas, passos de ferramentas, um diff real e os arquivos que a rodada alterou" />
</picture>

<sub>Modo Code: o agente encontrou a correção no projeto original, corrigiu o parser, adicionou um teste e rodou a suíte — com um modelo rodando no mesmo notebook.</sub>

</div>

---

## Conteúdo

[Chat](#chat) · [Code](#code) · [Imagem](#imagem) · [Documentos e pesquisa](#documentos-e-pesquisa) · [Voz](#voz) · [Feito para modelos pequenos](#feito-para-o-que-modelos-pequenos-erram) · [Modelos](#modelos) · [Privacidade](#privacidade) · [Instalação](#instalação) · [Compilação](#compilação) · [Arquitetura](#arquitetura)

## Chat

Um chat que mostra o raciocínio. O pensamento aparece num painel que você pode recolher; em modelos com níveis de esforço — low · medium · xhigh do Qwen3.8, os quatro do Muse-Glimmer, os do K2 Horizon — você escolhe o quanto ele pensa, mensagem a mensagem.

<picture>
  <source media="(prefers-color-scheme: light)" srcset="docs/screenshots/shot-chat-light.jpg" />
  <img src="docs/screenshots/shot-chat.jpg" width="860" alt="Uma resposta com código Rust destacado, uma tabela e uma fórmula KaTeX" />
</picture>

- **Renderizado enquanto chega** — código destacado, tabelas, KaTeX, Mermaid e HTML que roda ali mesmo. Os blocos são renderizados um a um, então um modelo rápido não faz a janela travar.
- **O que está ligado, num relance** — uma linha discreta acima da caixa de mensagem diz o que a próxima mensagem vai usar (raciocínio, busca na web, base de conhecimento), cada item a um clique de desligar.
- **Imagens também** — qualquer modelo com visão: um GGUF com seu projetor, ou uma versão MLX.
- **Canvas** — peça uma página e veja-a ganhar forma numa prévia ao vivo ao lado do código; as mudanças seguintes entram como patches.
- **Busca na web sem chave**, busca em texto completo no histórico e exportação para Markdown ou JSON.
- **Do seu jeito** — escuro quente ou frio, claro papel ou creme, quatro temas de código, English · 简体中文 · Português.

## Code

Um agente de programação para o modelo que você consegue rodar. Mude para **Code**, abra uma pasta e descreva a tarefa. O agente lê, busca, edita e roda comandos no seu projeto — cada passo aparece na hora, cada edição como um diff real — e fecha a rodada com um cartão listando cada arquivo alterado, cada um reversível.

- **Você no controle** — um plano de tarefas ao vivo; edições e comandos esperam sua aprovação, a menos que você permita. O acesso a arquivos fica restrito ao workspace; no macOS o shell roda na sandbox do sistema.
- **Comandos que continuam rodando** — servidores de desenvolvimento e watchers vão para segundo plano; um comando que faz uma pergunta (`[y/N]`, `Password:`, um REPL, o menu de um scaffolder) ganha um terminal para responder.
- **Ele lembra** — memória do projeto em Markdown simples; ele pode buscar nas próprias sessões anteriores, e você pode mencionar uma com @ para trazê-la.
- **Ferramentas que você adiciona** — servidores MCP (com uma lista curada e testada ao vivo), skills escritas em Markdown e um navegador que ele consegue controlar.
- **Nada concluído no vermelho** — a rodada termina num portão: o que mudou desde a última execução aprovada é verificado antes.

**Medido.** O mesmo modelo local em todas as linhas — Qwen3.5-35B-A3B (MoE, ~3 B ativos), mxfp8 no MLX, raciocínio desligado, uma só máquina:

| SWE-bench Verified, subconjunto de 45 tarefas validado no macOS | Resolvidas |
| --- | --- |
| **Agente do Chaty** (v1.9, contexto de 16K) | **15/45 (33 %)** |
| qwen-code 0.20 — a CLI da própria família do modelo (precisa de 32K) | 12/45 (27 %) |
| pi 0.81 — agente mínimo de 4 ferramentas | 10/45 (22 %) |
| opencode 1.18 | 7/45 (16 %) |
| agente só com bash — ablação de uma ferramenta | 6/45 (13 %) |

É um subconjunto rodado no macOS, então não se compara com os números de leaderboard; método, configurações e ressalvas em [docs/BENCHMARKS.md](docs/BENCHMARKS.md).

## Imagem

Carregue um modelo de texto para imagem e o app inteiro vira um estúdio: sessões que se leem como conversas, uma prévia ao vivo enquanto desenha e as configurações recomendadas de cada modelo já preenchidas.

<table>
<tr>
<td width="50%"><img src="docs/screenshots/plate-diner.jpg" alt="Um letreiro de neon escrito LATE NIGHT DINER numa esquina chuvosa" /></td>
<td width="50%"><img src="docs/screenshots/plate-shokudo.jpg" alt="Um letreiro de neon vermelho escrito 深夜食堂 numa rua chuvosa à noite" /></td>
</tr>
<tr>
<td><sub>“A neon sign that reads LATE NIGHT DINER on a rainy street corner…” — Z-Image Turbo, Q4_K_M, 1024², 8 passos, 2 min 51 s</sub></td>
<td><sub>“雨夜街角的霓虹灯招牌，写着「深夜食堂」…” — Qwen-Image 2.1, Q4_K_M, 1024², 20 passos, 16 min 10 s</sub></td>
</tr>
</table>

<sub>Direto do Chaty, sem edição, num Apple M4 Pro com 48 GB.</sub>

- **As famílias que importam** — Z-Image e Z-Image Turbo, Qwen-Image 2.1, FLUX.1 dev e schnell, Chroma, Stable Diffusion 1.x, 2.x, XL e 3.x.
- **Os outros arquivos, encontrados** — um GGUF de texto para imagem contém só o denoiser; o Chaty encontra o VAE e o codificador de texto ao lado dele, ou baixa os que faltam com um clique.
- **Continue a partir de uma imagem** — comece a próxima rodada de qualquer imagem; modelos que editam a usam como referência e mudam só o que você pedir.
- **Salva com a receita** — PNGs levam o prompt e as configurações dentro, organizados por data.

Roda no [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) num processo auxiliar — Metal no Mac, Vulkan no Windows e no Linux — então uma falha de driver encerra o auxiliar, não o app.

## Documentos e pesquisa

Arraste PDFs, arquivos do Word, apresentações, planilhas, Markdown ou código. O Chaty os indexa na sua máquina, busca por significado e por palavra-chave e cita o trecho por trás de cada afirmação — ou diz claramente quando seus documentos não cobrem a pergunta.

- **Digitalizações, lidas página a página** por um modelo com visão; gráficos dentro dos documentos podem ser descritos para também serem encontrados.
- **Deep Research** — várias rodadas de busca na web e raciocínio, num relatório que cita só o que usou; exporte para PDF ou Markdown.
- **Um podcast com dois apresentadores** a partir de uma base de conhecimento, salvo em WAV.

## Voz

O modo Live é uma conversa sem as mãos: o Whisper escuta, o modelo responde e uma voz local lê a resposta frase por frase — em inglês ou chinês. A fala roda na CPU, então nunca disputa a memória da GPU com o modelo. O silêncio envia sua vez; qualquer resposta pode ser lida em voz alta.

## Feito para o que modelos pequenos erram

Um modelo de fronteira segura um fluxo de trabalho sozinho. Um modelo que cabe no seu notebook muitas vezes não: repete a chamada que acabou de fazer, envia um argumento vazio, copia a linha que quer mudar um pouco errada e dá o trabalho por feito sem rodar nada. O Chaty foi construído para esse modelo.

| | |
|---|---|
| **O dialeto de cada um** | Chamadas de ferramenta são ensinadas e lidas no formato em que cada modelo foi treinado — XML, JSON, Gemma, LFM, K2, GLM, MiniCPM — tirado do template de chat dele, não presumido. |
| **Edições que pegam** | Uma edição é conferida contra o arquivo enquanto ainda está sendo escrita. Uma linha copiada com espaços, escapes ou números de linha errados ainda entra quando exatamente um lugar combina. |
| **Deslizes pegos no passo** | Chamadas repetidas, argumentos vazios e planos não executados são notados onde acontecem, com uma correção que mostra ao modelo o que ele escreveu. |
| **Nada concluído no vermelho** | O que mudou desde a última execução aprovada é verificado antes de a rodada poder terminar. |
| **Um cache que continua** | O prompt de cada rodada é um acréscimo ao anterior, então o cache do modelo é reaproveitado em vez de lido de novo desde o início. |
| **Um app só** | Sem servidor, sem porta, sem chave de API, sem arquivo de configuração. |

## Modelos

Qualquer GGUF roda no llama.cpp (Metal, ou Vulkan em NVIDIA, AMD e Intel); no Apple Silicon, pastas MLX também rodam nativamente. Busque e baixe do Hugging Face sem sair do app — ou aponte o Chaty para uma pasta que você já tem. Estas famílias têm templates, controles de raciocínio e formatos de chamada de ferramenta integrados:

| Família | O que está integrado |
|---|---|
| Qwen 3 · 3.5 · 3.6 · 3.8 | Raciocínio liga/desliga; níveis de esforço do Qwen3.8; visão quando o modelo tem |
| Gemma 3 · 4 | Visão; o formato de chamada e o raciocínio próprios do Gemma 4 |
| K2 Horizon · MoVA | Seus níveis de esforço e chamadas `<ifm\|arg_key>`, nos dois motores |
| GLM-4.5 · 4.6 · 4.7 | Suas chamadas `<tool_call>nome<arg_key>…` |
| Llama 3 · Muse-Glimmer | Visão, e os quatro níveis de esforço do Muse-Glimmer |
| MiniCPM5 · LFM 2.5 | O formato de chamada de cada família |
| [Chaty · Qwen3.5-4B design](https://huggingface.co/stevenpr/chaty-qwen3.5-4b-design-GGUF) | Nosso próprio ajuste fino para páginas web de um arquivo — um clique na primeira execução |

Um ajuste fino da comunidade cujo template de chat difere do palpite embutido do llama.cpp roda com o próprio template, como o transformers o rodaria.

## Privacidade

Modelos, conversas, documentos e imagens ficam numa pasta no seu disco. Não há conta nem servidor nosso em que confiar — apague a pasta e não sobra nada.

A rede é usada só para: **busca na web e Deep Research**, quando você as liga; **downloads que você inicia** (modelos, vozes, arquivos de embedding); e **uma verificação de atualização** — um pedido ao GitHub pela versão mais recente, alguns segundos após abrir. Detalhes em [Privacidade e dados](https://chaty.ca/docs.html#privacy).

## Instalação

Baixe da [versão mais recente](../../releases/latest):

| Plataforma | Arquivo | Observações |
|---|---|---|
| macOS (Apple Silicon) | `Chaty_*_aarch64.dmg` | Metal e MLX. Veja a nota da primeira execução abaixo |
| Windows 10 / 11 (x64) | `Chaty_*_x64-setup.exe` | Vulkan. Instalador por usuário, sem precisar de admin |
| Linux (x86-64) | `Chaty_*_amd64.AppImage` | Vulkan. Beta — `chmod +x` e execute |

**Primeira execução no macOS.** O Chaty é assinado, mas não notarizado (não há uma conta paga de desenvolvedor Apple por trás), então o Gatekeeper avisa na primeira abertura. Limpe a quarentena do download uma vez e abra o Chaty normalmente:

```sh
xattr -dr com.apple.quarantine /Applications/Chaty.app
```

Ou abra, dispense o aviso e escolha **Ajustes do Sistema → Privacidade e Segurança → Abrir Mesmo Assim**.

Modelos de linguagem pequenos e quantizados rodam em 8 GB de memória; modelos de texto para imagem pedem 16 GB ou mais. O Chaty dimensiona o uso da GPU à sua memória e recusa um modelo que não cabe, em vez de travar a máquina. Primeira vez? [Primeiros passos](https://chaty.ca/docs.html#getting-started).

## Compilação

Detalhes completos em **[BUILD.md](BUILD.md)**.

```bash
# macOS (Apple Silicon)
npm install
npm run tauri dev      # desenvolvimento (Metal)
npm run tauri build    # → .app + .dmg
```

```powershell
# Windows
npm install
.\dev.ps1                            # desenvolvimento
npm run tauri build -- --no-bundle   # exe de release → compile o instalador Inno
```

Os motores MLX e de imagem são auxiliares separados — `scripts/build-mlx-sidecar.sh` e `scripts/build-sd-sidecar.{sh,ps1}`. As versões saem do CI: ajuste a versão com `scripts/bump-version.sh x.y.z`, envie uma tag `vx.y.z` e o GitHub Actions compila as três plataformas numa só release.

## Arquitetura

| Camada | Stack |
|---|---|
| Shell | Tauri 2 — bandeja, atalho global, instância única |
| Interface | React 19 · Vite · react-markdown · KaTeX · Mermaid |
| Modelos de linguagem | Rust · `llama-cpp-2` (llama.cpp — Metal / Vulkan) · MLX por um auxiliar `mlx-swift-lm` no Apple Silicon · templates de chat renderizados com minijinja quando o do modelo difere |
| Modelos de imagem | stable-diffusion.cpp no auxiliar `chaty-sd` |
| Voz | `sherpa-rs` (ONNX Runtime, CPU) — Whisper, Kokoro-82M e uma voz chinesa VITS |
| Base de conhecimento | Embeddings bge-m3 + BM25 · busca híbrida RRF / MMR · armazenamento vetorial em SQLite |
| Armazenamento | SQLite — conversas, sessões, busca em texto completo |

## Contribuindo

Relatos de bugs são bem-vindos — anexar o log de erros (**Configurações → Dados → Abrir log de erro**) costuma transformar dias de palpites em minutos. Veja [Contribuindo](https://chaty.ca/docs.html#contributing) para compilar, testar e rodar os benchmarks.

## Licença

MIT — veja [LICENSE](LICENSE). Construído sobre [llama.cpp](https://github.com/ggml-org/llama.cpp), [MLX](https://github.com/ml-explore/mlx-swift), [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp), [Tauri](https://tauri.app) e [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx).
