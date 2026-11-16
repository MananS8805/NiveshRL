# Register (or remove) a Windows scheduled task that opens NiveshRL in the tray at 09:00, Monday-Friday, so the
# intraday copilot runs every market day by itself: 09:05 morning brief -> paper trading from 09:20 -> 15:20 day review.
#
#   powershell -ExecutionPolicy Bypass -File scripts\schedule_copilot.ps1            # register
#   powershell -ExecutionPolicy Bypass -File scripts\schedule_copilot.ps1 -Remove    # unregister
#
# The task runs as you, only while you are logged on, and may wake the laptop from sleep (if Windows allows wake timers;
# it cannot start a shut-down laptop). If NiveshRL is already running it does nothing. Paper only: nothing places orders.
param([switch]$Remove, [string]$Time = "09:00", [string]$Exe = "$env:USERPROFILE\Desktop\NiveshRL\NiveshRL.exe")

$TaskName = "NiveshRL copilot (market days)"

if ($Remove) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
    Write-Host "Removed '$TaskName'."
    exit 0
}
if (-not (Test-Path $Exe)) { Write-Error "NiveshRL.exe not found at $Exe (pass -Exe <path>)."; exit 1 }

$Action = New-ScheduledTaskAction -Execute $Exe -Argument "--tray" -WorkingDirectory (Split-Path -Parent $Exe)
$Trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday, Tuesday, Wednesday, Thursday, Friday -At $Time
$Settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -WakeToRun -DontStopIfGoingOnBatteries -AllowStartIfOnBatteries `
    -ExecutionTimeLimit (New-TimeSpan -Hours 0) -MultipleInstances IgnoreNew
Register-ScheduledTask -TaskName $TaskName -Action $Action -Trigger $Trigger -Settings $Settings `
    -Description "Opens NiveshRL in the tray on weekdays so the intraday copilot runs (paper only)" -Force | Out-Null
Write-Host "Registered '$TaskName' for $Time Mon-Fri -> $Exe --tray"
Write-Host "Keep 'Run automatically every weekday' ticked on the Intraday copilot screen (it is on now)."
