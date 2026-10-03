; Inno Setup 6 script for NiveshRL-Setup.exe. Built by packaging\build.ps1 -Installer
; (expects the PyInstaller output in build\dist\NiveshRL).
; Per-user install (no admin rights needed); user data lives in %LOCALAPPDATA%\NiveshRL
; and is left in place on uninstall.

#define AppName "NiveshRL"
#define AppVersion "1.0.0"

[Setup]
AppId={{6E3B7C1A-4F2D-4C8B-9A51-2B7E0D9C4A11}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher=NiveshRL (educational project)
DefaultDirName={localappdata}\Programs\{#AppName}
DefaultGroupName={#AppName}
PrivilegesRequired=lowest
OutputDir=..\build\installer
OutputBaseFilename=NiveshRL-Setup
SetupIconFile=niveshrl.ico
UninstallDisplayIcon={app}\NiveshRL.exe
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
ArchitecturesInstallIn64BitMode=x64compatible
ArchitecturesAllowed=x64compatible

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Shortcuts:"
Name: "autostart"; Description: "Start NiveshRL in the tray when Windows starts (keeps live prices, alerts and the 4 PM refresh running)"; GroupDescription: "Background:"; Flags: unchecked

[Files]
Source: "..\build\dist\NiveshRL\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\NiveshRL.exe"
Name: "{group}\Uninstall {#AppName}"; Filename: "{uninstallexe}"
Name: "{userdesktop}\{#AppName}"; Filename: "{app}\NiveshRL.exe"; Tasks: desktopicon

[Registry]
Root: HKCU; Subkey: "Software\Microsoft\Windows\CurrentVersion\Run"; ValueType: string; ValueName: "NiveshRL"; ValueData: """{app}\NiveshRL.exe"" --tray"; Flags: uninsdeletevalue; Tasks: autostart

[Run]
Filename: "{app}\NiveshRL.exe"; Description: "Launch NiveshRL"; Flags: nowait postinstall skipifsilent

[UninstallRun]
Filename: "{cmd}"; Parameters: "/c taskkill /im NiveshRL.exe /f"; Flags: runhidden; RunOnceId: "KillApp"

[Messages]
WelcomeLabel2=This installs the NiveshRL trading desk: live NIFTY 200 monitor, daily briefing, screener, watchlist with alerts, news sentiment (FinBERT) and deep-learning research models.%n%nEducational project, not investment advice. Not registered with SEBI.
