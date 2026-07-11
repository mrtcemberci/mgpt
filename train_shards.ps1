<#
.SYNOPSIS
    Automated multi-shard training script for MGPT on Windows PowerShell.
    Loops through all .txt files in a directory and saves each shard's training run
    to a separate checkpoint file (tinystories_shard00.bin, tinystories_shard01.bin, etc.).

.USAGE
    .\train_shards.ps1 [-DataDir "tinystories"] [-StepsPerShard 3000] [-TotalSteps 72000] [-LR 0.0003]
#>

param(
    [string]$DataDir = "tinystories",
    [string]$MasterVocab = "master_vocab.bin",
    [int]$StepsPerShard = 3000,
    [int]$TotalSteps = 72000,
    [float]$LR = 0.0003,
    [int]$Layers = 12,
    [int]$Channels = 384,
    [int]$Window = 256,
    [int]$Batch = 16,
    [int]$Accumulate = 2,
    [string]$ExePath = ".\build_release\Release\mgpt.exe"
)

# Ensure executable exists
if (-not (Test-Path $ExePath)) {
    Write-Error "MGPT executable not found at '$ExePath'. Please build the project first."
    exit 1
}

# Ensure data directory exists
if (-not (Test-Path $DataDir)) {
    Write-Error "Data directory '$DataDir' does not exist."
    exit 1
}

# Get all .txt files sorted alphabetically
$shards = Get-ChildItem -Path $DataDir -Filter "*.txt" | Sort-Object Name

if ($shards.Count -eq 0) {
    Write-Error "No .txt shard files found in '$DataDir'."
    exit 1
}

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "   MGPT MULTI-SHARD AUTOMATED PIPELINE (SHARD RECOVERY)    " -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Found $($shards.Count) shards in '$DataDir'" -ForegroundColor Green
Write-Host "Model Config: ${Layers}L, ${Channels}C, ${Window}W | Batch: $Batch (Accum: $Accumulate) | LR: $LR" -ForegroundColor Green
Write-Host "============================================================`n" -ForegroundColor Cyan

for ($i = 0; $i -lt $shards.Count; $i++) {
    $shardFile = $shards[$i].FullName
    $shardName = $shards[$i].Name
    $idxStr = "{0:D2}" -f $i

    $currentBin = "tinystories_shard${idxStr}.bin"
    $prevBin = "tinystories_shard{0:D2}.bin" -f ($i - 1)

    Write-Host "------------------------------------------------------------" -ForegroundColor Yellow
    Write-Host "[Shard $($i + 1)/$($shards.Count)] Processing: $shardName -> Saving to: $currentBin" -ForegroundColor Yellow
    Write-Host "------------------------------------------------------------" -ForegroundColor Yellow

    $cmdArgs = @(
        "-t", "-g",
        "-d=`"$shardFile`"",
        "-f=`"$currentBin`"",
        "--lr=$LR",
        "-s=$StepsPerShard",
        "--total-steps=$TotalSteps",
        "-l=$Layers",
        "-c=$Channels",
        "-w=$Window",
        "-b=$Batch",
        "-a=$Accumulate"
    )

    if ($MasterVocab -ne "") {
        $cmdArgs += "--vocab-file=`"$MasterVocab`""
    }

    if ($i -gt 0) {
        # For shard 1 and above, copy the previous shard's checkpoint if current checkpoint doesn't exist yet
        if (-not (Test-Path $currentBin)) {
            if (-not (Test-Path $prevBin)) {
                Write-Error "Previous shard checkpoint '$prevBin' missing! Cannot resume shard $idxStr."
                exit 1
            }
            Write-Host "Copying previous checkpoint ($prevBin -> $currentBin) to resume state..." -ForegroundColor Cyan
            Copy-Item -Path $prevBin -Destination $currentBin -Force
        }
        $cmdArgs += "--resume"
    }

    Write-Host "Running: $ExePath $($cmdArgs -join ' ')" -ForegroundColor DarkGray
    & $ExePath $cmdArgs

    if ($LASTEXITCODE -ne 0) {
        Write-Error "Training failed on shard $shardName (Exit Code: $LASTEXITCODE). Pipeline stopped."
        exit $LASTEXITCODE
    }

    Write-Host "`nSuccessfully completed shard $idxStr. Saved recovery checkpoint: $currentBin`n" -ForegroundColor Green
}

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "   ALL $($shards.Count) SHARDS COMPLETED SUCCESSFULLY!     " -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Cyan
