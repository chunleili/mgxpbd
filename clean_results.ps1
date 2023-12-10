# remove-item except .gitignore
remove-item -path ".\results\*" -Recurse -exclude .gitignore
Write-Output "Cleaned results folder"
