tell application "Notes"
    set exportFolder to (POSIX path of (path to desktop)) & "notes_export/"
    do shell script "mkdir -p " & quoted form of exportFolder

    -- Record export start time BEFORE the loop.
    -- The sentinel is written as (start - 10s) so any note modified during
    -- the export window is still captured on the next run.
    set sentinelPath to exportFolder & ".last_export"
    set exportStartEpoch to (do shell script "date +%s") as integer

    -- Read last export timestamp (Unix epoch). Default 0 = export everything.
    set cutoffEpoch to 0
    try
        set cutoffEpoch to (do shell script "cat " & quoted form of sentinelPath) as integer
    end try

    -- AppleScript dates are seconds since 2001-01-01; Unix epoch is 978307200s earlier
    set refDate to date "Thursday, January 1, 2001 at 12:00:00 AM"
    set asOffset to 978307200

    set allNotes to every note
    set manifestContent to ""
    set exportCount to 0
    set noteCount to 0

    repeat with theNote in allNotes
        set noteCount to noteCount + 1
        set noteName to name of theNote
        set noteModEpoch to ((modification date of theNote) - refDate) + asOffset

        -- Extract stable short ID from "x-coredata://UUID/ICNote/pNNN"
        set rawId to id of theNote
        set AppleScript's text item delimiters to "/"
        set idParts to text items of rawId
        set shortId to last item of idParts
        set AppleScript's text item delimiters to ""

        set filePath to exportFolder & shortId & ".html"

        -- Only fetch body + write if note changed since last export
        if noteModEpoch > cutoffEpoch then
            try
                set noteBody to body of theNote
                set fileRef to open for access filePath with write permission
                set eof of fileRef to 0
                write noteBody to fileRef
                close access fileRef
                set exportCount to exportCount + 1
            end try
        end if

        set manifestContent to manifestContent & shortId & "|||" & noteName & "|||" & noteModEpoch & linefeed
    end repeat

    -- Write manifest in one shot
    do shell script "rm -f " & quoted form of (exportFolder & "manifest.txt")
    set mfRef to open for access (exportFolder & "manifest.txt") with write permission
    write manifestContent to mfRef
    close access mfRef

    -- Write sentinel as (start - 10s): conservative overlap ensures notes
    -- edited during this export run are not missed on the next run.
    set sentinelEpoch to exportStartEpoch - 10
    do shell script "echo " & sentinelEpoch & " > " & quoted form of sentinelPath

    return "Exported " & exportCount & " of " & noteCount & " notes"
end tell
