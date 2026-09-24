; Inno Setup script for Chaty — a modern, per-user installer.
; Build:  ISCC /DAppVersion=x.y.z /DSrcDir=C:\path\to\release Chaty.iss
; (AppVersion + SrcDir can also be left to the defaults below.)

#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif
#ifndef SrcDir
  #define SrcDir "..\..\..\..\ct\release"
#endif

#define AppName "Chaty"
#define AppPublisher "Fangyuan Lin"
#define AppURL "https://github.com/Fangyuan025/Chaty"
#define AppExe "chaty.exe"

[Setup]
AppId={{8F4C9E2A-3B7D-4E1C-9A6F-CHATY0DESKTOP}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
AppPublisherURL={#AppURL}
AppSupportURL={#AppURL}
VersionInfoVersion={#AppVersion}
DefaultDirName={localappdata}\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
DisableDirPage=auto
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir={#SrcDir}\bundle\inno
OutputBaseFilename=Chaty_{#AppVersion}_x64-setup
SetupIconFile=..\icons\icon.ico
UninstallDisplayIcon={app}\{#AppExe}
UninstallDisplayName={#AppName}
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
WizardSizePercent=100
ShowLanguageDialog=no
; Pick the language afresh on every run. The default (yes) reuses the
; language a previous install recorded in the registry — and every install
; before Chinese existed recorded English, so upgrading over one stayed
; English on a Chinese Windows (issue #16, after the first fix). Nobody ever
; chose that language: there was no other.
UsePreviousLanguage=no
CloseApplications=yes

[Languages]
; Setup speaks the Windows UI language when it has it (LanguageDetectionMethod
; defaults to uilanguage) and falls back to the first entry, English — no
; language dialog either way (ShowLanguageDialog=no). Issue #16: a Chinese
; Windows got an all-English installer.
Name: "english"; MessagesFile: "compiler:Default.isl"
; Chinese Simplified is an official Inno Setup translation (6.5+): the
; compiler's own copy when it ships one, else the copy kept beside this
; script (Languages\ChineseSimplified.isl, from jrsoftware/issrc).
#if FileExists(AddBackslash(CompilerPath) + "Languages\ChineseSimplified.isl")
Name: "chinesesimplified"; MessagesFile: "compiler:Languages\ChineseSimplified.isl"
#else
Name: "chinesesimplified"; MessagesFile: "Languages\ChineseSimplified.isl"
#endif

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Files]
Source: "{#SrcDir}\{#AppExe}"; DestDir: "{app}"; Flags: ignoreversion
; The image engine (stable-diffusion.cpp sidecar) — text-to-image models run
; in it. tauri-build copies it next to the exe from src-tauri/binaries.
Source: "{#SrcDir}\chaty-sd.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#SrcDir}\onnxruntime.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#SrcDir}\onnxruntime_providers_shared.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#SrcDir}\sherpa-onnx-c-api.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#SrcDir}\sherpa-onnx-cxx-api.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#SrcDir}\cargs.dll"; DestDir: "{app}"; Flags: ignoreversion
; Vulkan runtime loader: normally provided by the GPU driver, but bare
; machines/VMs (e.g. winget validation) lack it and the app fails to start.
; CI copies it from the Vulkan SDK next to the exe before ISCC runs; the local
; loader wins only when the system has none (standard loader search order).
Source: "{#SrcDir}\vulkan-1.dll"; DestDir: "{app}"; Flags: ignoreversion skipifsourcedoesntexist

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\{#AppExe}"
Name: "{group}\{cm:UninstallProgram,{#AppName}}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExe}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExe}"; Description: "{cm:LaunchProgram,{#AppName}}"; Flags: nowait postinstall skipifsilent

[Code]
{ Ensure the WebView2 runtime (required by the Tauri WebView) is present. }
function IsWebView2Installed: Boolean;
var
  v: String;
  g: String;
begin
  g := '{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}';
  Result :=
    RegQueryStringValue(HKLM, 'SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate\Clients\' + g, 'pv', v) or
    RegQueryStringValue(HKLM, 'SOFTWARE\Microsoft\EdgeUpdate\Clients\' + g, 'pv', v) or
    RegQueryStringValue(HKCU, 'SOFTWARE\Microsoft\EdgeUpdate\Clients\' + g, 'pv', v);
  if Result then
    Result := (v <> '') and (v <> '0.0.0.0');
end;

procedure CurStepChanged(CurStep: TSetupStep);
var
  code: Integer;
begin
  if (CurStep = ssInstall) and (not IsWebView2Installed) then
  begin
    try
      DownloadTemporaryFile(
        'https://go.microsoft.com/fwlink/p/?LinkId=2124703',
        'MicrosoftEdgeWebview2Setup.exe', '', nil);
      Exec(ExpandConstant('{tmp}\MicrosoftEdgeWebview2Setup.exe'),
        '/silent /install', '', SW_HIDE, ewWaitUntilTerminated, code);
    except
      { Non-fatal: most up-to-date Windows already ships the WebView2 runtime. }
    end;
  end;
end;
