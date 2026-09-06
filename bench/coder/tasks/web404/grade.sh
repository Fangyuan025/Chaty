#!/bin/bash
# The recovery, not the retrying: the content must have come from the URL that
# works, and it must have been written down.
set -e
test -f found.txt
grep -q "简体中文" found.txt
grep -qE "404|不存在|does not exist|not found|Not Found" NOTES.md
