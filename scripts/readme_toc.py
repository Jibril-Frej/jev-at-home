"""Rewrite the table of contents in README.md between <!-- toc --> and <!-- tocstop -->.

Run automatically by the git pre-commit hook (see the script's install note below).
Lists every ## to #### heading (the # title is skipped), with GitHub's anchor links.
Run: python3 scripts/readme_toc.py [README.md]
"""
import re
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "README.md"
text = open(path, encoding="utf-8").read()

lines, seen, fence = [], {}, False
for line in text.splitlines():
    if line.lstrip().startswith("```"):
        fence = not fence
        continue
    m = None if fence else re.match(r"^(#{2,4})\s+(.+?)\s*#*\s*$", line)
    if not m:
        continue
    title = m.group(2)
    # GitHub anchors: lowercase, drop punctuation except - and _, spaces to -
    slug = re.sub(r"[^\w\- ]", "", title.lower()).replace(" ", "-")
    n = seen.get(slug, 0)
    seen[slug] = n + 1
    anchor = slug if n == 0 else f"{slug}-{n}"
    lines.append(f"{'  ' * (len(m.group(1)) - 2)}- [{title}](#{anchor})")

toc = "<!-- toc -->\n\n" + "\n".join(lines) + "\n\n<!-- tocstop -->"
new = re.sub(r"<!-- toc -->.*?<!-- tocstop -->", lambda _: toc, text, flags=re.S)
if new != text:
    open(path, "w", encoding="utf-8").write(new)

# Install as a pre-commit hook (once per clone), so it runs whenever README.md is committed:
#   cat > .git/hooks/pre-commit <<'HOOK'
#   #!/bin/sh
#   if git diff --cached --name-only | grep -qx README.md; then
#     python3 scripts/readme_toc.py && git add README.md
#   fi
#   HOOK
#   chmod +x .git/hooks/pre-commit
