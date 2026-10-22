$ErrorActionPreference = "Stop"

$recordingId = "bb8f1d9e-b84b-4ff2-a92e-f1a378bf67d4"
$dataDirectory = Join-Path $PSScriptRoot "data"
$downloads = @(
    @{
        Name = "Athirum Kazhal.mp3"
        Url  = "https://dunya.compmusic.upf.edu/document/by-id/$recordingId/mp3"
    },
    @{
        Name = "Athirum Kazhal.multitrack-ghatam.mp3"
        Url  = "https://dunya.compmusic.upf.edu/document/by-id/$recordingId/multitrack-ghatam"
    }
)
$temporaryFiles = [System.Collections.Generic.List[string]]::new()
$backupFiles = [System.Collections.Generic.List[string]]::new()
$tokenSecure = $null
$tokenPointer = [IntPtr]::Zero
$token = $null
$client = $null
$stagedDownloads = @()

function Save-SaragaFile {
    param(
        [System.Net.Http.HttpClient] $Client,
        [string] $Url,
        [string] $Destination
    )

    $request = [System.Net.Http.HttpRequestMessage]::new(
        [System.Net.Http.HttpMethod]::Get,
        [Uri]$Url
    )
    $response = $null
    $sourceStream = $null
    $fileStream = $null

    try {
        $response = $Client.SendAsync(
            $request,
            [System.Net.Http.HttpCompletionOption]::ResponseHeadersRead
        ).GetAwaiter().GetResult()

        if (-not $response.IsSuccessStatusCode) {
            throw "HTTP $([int]$response.StatusCode) $($response.ReasonPhrase) for $Url"
        }

        $sourceStream = $response.Content.ReadAsStreamAsync().GetAwaiter().GetResult()
        $fileStream = [System.IO.File]::Open(
            $Destination,
            [System.IO.FileMode]::CreateNew,
            [System.IO.FileAccess]::Write,
            [System.IO.FileShare]::None
        )
        $sourceStream.CopyToAsync($fileStream).GetAwaiter().GetResult()
        $fileStream.Dispose()
        $fileStream = $null

        $actualLength = (Get-Item -LiteralPath $Destination).Length
        if ($actualLength -le 0) {
            throw "Downloaded file is empty: $Destination"
        }

        $expectedLength = $response.Content.Headers.ContentLength
        if ($null -ne $expectedLength -and $actualLength -ne [long]$expectedLength) {
            throw "Incomplete download for $Url (expected $expectedLength bytes, received $actualLength)"
        }

        return [long]$actualLength
    }
    finally {
        if ($null -ne $fileStream) {
            $fileStream.Dispose()
        }
        if ($null -ne $sourceStream) {
            $sourceStream.Dispose()
        }
        if ($null -ne $response) {
            $response.Dispose()
        }
        $request.Dispose()
    }
}

try {
    if (-not (Test-Path -LiteralPath $dataDirectory -PathType Container)) {
        throw "Project data folder not found: $dataDirectory"
    }

    $stagedDownloads = @(
        foreach ($download in $downloads) {
            $destination = Join-Path $dataDirectory $download.Name
            [PSCustomObject]@{
                Name           = $download.Name
                Url            = $download.Url
                Destination    = $destination
                TemporaryPath  = "$destination.$([Guid]::NewGuid().ToString('N')).part"
                BackupPath     = $null
                Length         = 0L
                OriginalExisted = Test-Path -LiteralPath $destination
                BackupCreated  = $false
                NewPlaced      = $false
            }
        }
    )

    foreach ($download in $stagedDownloads) {
        if ($download.OriginalExisted -and -not (Test-Path -LiteralPath $download.Destination -PathType Leaf)) {
            throw "Destination exists but is not a file: $($download.Destination)"
        }
    }

    $existingFiles = @($stagedDownloads | Where-Object { $_.OriginalExisted })
    if ($existingFiles.Count -gt 0) {
        Write-Host "These destination files already exist:"
        $existingFiles | ForEach-Object { Write-Host "  $($_.Destination)" }
        $confirmation = Read-Host 'Type OVERWRITE to replace them (originals will be kept as .backup files)'
        if ($confirmation -cne "OVERWRITE") {
            throw "Overwrite was not confirmed; no files were changed and no downloads were started."
        }
    }

    $tokenSecure = Read-Host "Enter your Dunya API token" -AsSecureString
    if ($tokenSecure.Length -eq 0) {
        throw "The Dunya API token cannot be empty."
    }

    $tokenPointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($tokenSecure)
    $token = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($tokenPointer)

    $client = [System.Net.Http.HttpClient]::new()
    $client.Timeout = [TimeSpan]::FromMinutes(10)
    $client.DefaultRequestHeaders.Authorization =
        [System.Net.Http.Headers.AuthenticationHeaderValue]::new("Token", $token)

    foreach ($download in $stagedDownloads) {
        $temporaryFiles.Add($download.TemporaryPath)
        $download.Length = Save-SaragaFile -Client $client -Url $download.Url -Destination $download.TemporaryPath
    }

    try {
        foreach ($download in $stagedDownloads) {
            if ($download.OriginalExisted) {
                do {
                    $download.BackupPath = "$($download.Destination).$([Guid]::NewGuid().ToString('N')).backup"
                } while (Test-Path -LiteralPath $download.BackupPath)

                Move-Item -LiteralPath $download.Destination -Destination $download.BackupPath
                $download.BackupCreated = $true
                $backupFiles.Add($download.BackupPath)
            }

            Move-Item -LiteralPath $download.TemporaryPath -Destination $download.Destination
            $download.NewPlaced = $true
        }
    }
    catch {
        $placementError = $_.Exception.Message
        $rollbackErrors = [System.Collections.Generic.List[string]]::new()
        $rollbackOrder = @($stagedDownloads)
        [Array]::Reverse($rollbackOrder)

        foreach ($download in $rollbackOrder) {
            if ($download.NewPlaced -and (Test-Path -LiteralPath $download.Destination -PathType Leaf)) {
                try {
                    Move-Item -LiteralPath $download.Destination -Destination $download.TemporaryPath
                    $download.NewPlaced = $false
                }
                catch {
                    $rollbackErrors.Add("Could not move new file '$($download.Destination)' back to staging: $($_.Exception.Message)")
                }
            }

            if ($download.BackupCreated -and (Test-Path -LiteralPath $download.BackupPath -PathType Leaf)) {
                if (-not (Test-Path -LiteralPath $download.Destination)) {
                    try {
                        Move-Item -LiteralPath $download.BackupPath -Destination $download.Destination
                        $download.BackupCreated = $false
                    }
                    catch {
                        $rollbackErrors.Add("Could not restore original '$($download.Destination)'; backup remains at '$($download.BackupPath)': $($_.Exception.Message)")
                    }
                }
                else {
                    $rollbackErrors.Add("Could not restore original '$($download.Destination)' because that path is occupied; backup remains at '$($download.BackupPath)'.")
                }
            }
        }

        if ($rollbackErrors.Count -gt 0) {
            throw "Final placement failed: $placementError Rollback also had issues: $($rollbackErrors -join ' ')"
        }
        throw "Final placement failed and originals were restored where applicable: $placementError"
    }

    foreach ($download in $stagedDownloads) {
        Write-Host ("Downloaded {0} ({1:N0} bytes)" -f $download.Name, $download.Length)
    }
    Write-Host "Success: both Saraga audio files are saved in data\."
    if ($backupFiles.Count -gt 0) {
        Write-Host "Original files were preserved as:"
        $backupFiles | ForEach-Object { Write-Host "  $_" }
    }
    Write-Host "Validation checks HTTP status, non-empty content, and Content-Length when provided."
    Write-Host "These checks do not guarantee valid audio or detect every possible truncation."
}
catch {
    Write-Error "Saraga download failed: $($_.Exception.Message)" -ErrorAction Continue
    exit 1
}
finally {
    foreach ($temporaryPath in $temporaryFiles) {
        if (Test-Path -LiteralPath $temporaryPath -PathType Leaf) {
            try {
                Remove-Item -LiteralPath $temporaryPath -Force -ErrorAction Stop
            }
            catch {
                Write-Warning "Could not remove temporary file '$temporaryPath': $($_.Exception.Message)"
            }
        }
    }
    if ($null -ne $client) {
        try {
            $client.DefaultRequestHeaders.Authorization = $null
        }
        catch {
            Write-Warning "Could not clear the Dunya authorization header during cleanup."
        }
        finally {
            $client.Dispose()
        }
    }
    if ($tokenPointer -ne [IntPtr]::Zero) {
        [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($tokenPointer)
    }
    if ($null -ne $tokenSecure) {
        $tokenSecure.Dispose()
    }
    Remove-Variable token -ErrorAction SilentlyContinue
}
