# Pre-push human review checklist

`scripts/verify_release.sh` is the **automated** half of the
pre-publish review. This file is the **human** half — the things a
script can't catch reliably.

Run the automated check first:

```bash
bash scripts/verify_release.sh
```

If it exits 0, then walk through the items below before running
`git push origin main`. Tick each box. Skipping any of them is how
private content leaks into open source.

## ☑️ The 10 questions

### 1. Is the auto checker green?

```
$ bash scripts/verify_release.sh
✓ All 6 checks passed. Safe to git push.
```

### 2. Spotlight search the working directory for the project name

```bash
grep -ri "Baku" Sources Tests docs examples Benchmarks scripts
grep -ri "baku" Sources Tests docs examples Benchmarks scripts
```

Both should print **nothing**.

### 3. Spotlight search for any other proprietary names

Anything you typed at any point in the project: prompt fragments
("小袋鼠", "薰衣草"), character names ("甜心", "棉花"), API keys,
HuggingFace tokens, commercial product names. If it shouldn't be
public, it should not exist anywhere in the working tree.

### 4. List untracked files — should be empty

```bash
git status --short
```

If there's anything untracked, decide explicitly whether each item
should be added or `.gitignore`'d. Don't `git add -A` blindly.

### 5. Look at the file list one more time

```bash
git ls-files | sort
```

Read it line by line. Any file that doesn't belong in an open-source
package should jump out. Examples of things that should NOT be there:
`.xcodeproj` directories, `xcuserdata`, `*.mobileprovision`, anything
containing your Apple Developer Team ID, anything containing your
HuggingFace API token.

### 6. Check the git log authors

```bash
git log --format='%an <%ae>' | sort -u
```

Should be exactly **one** entry: `Jingyang Ye <yejingyang8963@gmail.com>`.
If you see a second entry that includes a username like
`john@JOHNs-Mac-Studio.local` or anything from a parent project, the
local git config is wrong — fix it before pushing:

```bash
git config user.name "Jingyang Ye"
git config user.email "yejingyang8963@gmail.com"
```

(Note: `git config` without `--global` only affects this repo.)

### 7. Verify the LICENSE has YOUR name

```bash
head -3 LICENSE
```

Should show:
```
MIT License

Copyright (c) 2026 Jingyang Ye
```

### 8. Verify the README header has YOUR name

```bash
grep -i "Jingyang Ye" README.md
```

Should match in at least 2 places (the byline near the top and the
"Author" section near the bottom).

### 9. Check the remote URL

```bash
git remote -v
```

Should show:
```
origin  git@github.com:yejingyang8963-byte/Swift-gemma4-core.git (fetch)
origin  git@github.com:yejingyang8963-byte/Swift-gemma4-core.git (push)
```

If it points anywhere else (especially the parent project's repo),
**stop**. Do not push.

### 10. Final visual scan

Open the repo in Finder. Look at it.

- Does the file count look right? (~50–60 files for v0.1.0)
- Does anything look out of place?
- Are there any ".DS_Store" files visible? (There shouldn't be —
  they're in .gitignore — but the check is free.)

If everything looks right and you've ticked every box above, you can
push:

```bash
git push -u origin main
```

If you have ANY hesitation about ANY item above, do not push. Open
an issue on the GitHub repo asking for a second pair of eyes, or
ping the original maintainer. Open source is one-way: once it's
out there, it's out there forever.

## After the first push

- Watch the GitHub Actions CI run for ~5 minutes. If anything fails,
  fix it before announcing the project anywhere.
- Visit your repo on GitHub. Confirm the README renders, the banner
  image displays, and the badges are showing the right colors.
- Check the "Insights → Community Standards" page. It should show
  green checkmarks for LICENSE, README, CoC, Contributing,
  Issue templates, PR templates, and Security policy.
- Tag `v0.1.0` and push the tag to trigger the Release workflow:

```bash
git tag -a v0.1.0 -m "v0.1.0 — initial release"
git push origin v0.1.0
```

The Release workflow will draft a GitHub Release. Read its body, edit
if needed, then publish it manually from the web UI.
