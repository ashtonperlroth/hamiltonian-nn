Stage all changes, write a concise commit message based on the diff, commit, and push:
```bash
git diff --stat
git diff --cached --stat
```

Review the changes above, then:
1. `git add -A`
2. Write a descriptive one-line commit message
3. `git commit -m "<message>"`
4. `git push`