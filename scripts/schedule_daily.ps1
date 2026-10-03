# Register (or remove) a Windows scheduled task that runs the NiveshRL daily pipeline
# at 16:00 local time, Monday-Friday, after the NSE close.
#
#   powershell -ExecutionPolicy Bypass -File scripts\schedule_daily.ps1            # register
#   powershell -ExecutionPolicy Bypass -File scripts\schedule_daily.ps1 -Remove    # unregister
#
# Only needed when the desktop app (which runs the pipeline itself from its tray icon)
# isn't running at 16:00. The task runs as the current user, only when logged on.
param([switch]$Remove, [string]$Time = "16:00")

$TaskName = "NiveshRL daily pipeline"
$Root = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Root ".venv\Scripts\python.exe"
$Script = Join-Path $Root "scripts\daily.py"
$Log = Join-Path $Root "data\daily\scheduled_run.log"

if ($Remove) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
    Write-Host "Removed '$TaskName'."
    exit 0
}
if (-not (Test-Path $Python)) { Write-Error "No venv at $Python. Run demo.bat first."; exit 1 }

$Action = New-ScheduledTaskAction -Execute "cmd.exe" `
    -Argument "/c `"`"$Python`" `"$Script`" >> `"$Log`" 2>&1`"" -WorkingDirectory $Root
$Trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday, Tuesday, Wednesday, Thursday, Friday -At $Time
$Settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -DontStopIfGoingOnBatteries `
    -ExecutionTimeLimit (New-TimeSpan -Hours 1) -MultipleInstances IgnoreNew
Register-ScheduledTask -TaskName $TaskName -Action $Action -Trigger $Trigger -Settings $Settings `
    -Description "NiveshRL: prices, fundamentals, news sentiment, next-day model, briefing" -Force | Out-Null
Write-Host "Registered '$TaskName' for $Time Mon-Fri. Log: $Log"
