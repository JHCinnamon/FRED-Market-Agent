Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing

$runnerDirectory = $PSScriptRoot
$runnerScript = Join-Path $runnerDirectory "run.cmd"
$iconPath = Join-Path $PSScriptRoot "GitHub-Runner.ico"

if (-not (Test-Path -LiteralPath $runnerScript -PathType Leaf)) {
    [System.Windows.Forms.MessageBox]::Show(
        "Could not find the GitHub Actions runner at:`n$runnerScript",
        "GitHub Runner",
        [System.Windows.Forms.MessageBoxButtons]::OK,
        [System.Windows.Forms.MessageBoxIcon]::Error
    )
    exit 1
}

if (-not (Test-Path -LiteralPath $iconPath -PathType Leaf)) {
    [System.Windows.Forms.MessageBox]::Show(
        "Could not find the tray icon at:`n$iconPath",
        "GitHub Runner",
        [System.Windows.Forms.MessageBoxButtons]::OK,
        [System.Windows.Forms.MessageBoxIcon]::Error
    )
    exit 1
}

$runnerProcess = Start-Process -FilePath $env:ComSpec -WorkingDirectory $runnerDirectory -ArgumentList "/k `"$runnerScript`"" -PassThru

$trayMenu = New-Object System.Windows.Forms.ContextMenuStrip
$exitItem = $trayMenu.Items.Add("Exit Runner")

$trayIcon = New-Object System.Windows.Forms.NotifyIcon
$trayIcon.ContextMenuStrip = $trayMenu
$trayIcon.Icon = New-Object System.Drawing.Icon($iconPath)
$trayIcon.Text = "GitHub Actions Runner"
$trayIcon.Visible = $true
$trayIcon.ShowBalloonTip(3000, "GitHub Actions Runner", "Runner window started. The tray menu can stop it.", [System.Windows.Forms.ToolTipIcon]::Info)

$applicationContext = New-Object System.Windows.Forms.ApplicationContext
$exitItem.add_Click({
    if (-not $runnerProcess.HasExited) {
        & taskkill.exe /PID $runnerProcess.Id /T /F | Out-Null
    }
    $trayIcon.Visible = $false
    $trayIcon.Dispose()
    $applicationContext.ExitThread()
})

[System.Windows.Forms.Application]::Run($applicationContext)