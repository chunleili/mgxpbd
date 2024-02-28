# remove-item except .gitignore
remove-item -path ".\result\*" -Recurse -exclude .gitignore
Write-Output "Cleaned result folder"
