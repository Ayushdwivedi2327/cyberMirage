"""
Cyber Mirage - Git Push Utility
Pushes the local commits to the remote repository.
Supports GitHub Personal Access Token (PAT) authentication.
"""

import sys
import os
import dulwich.porcelain as porcelain
from dulwich.repo import Repo

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REMOTE_URL = "https://github.com/Ayushdwivedi2327/cyberMirage.git"


def push_repo(token: str = None):
    repo = Repo(REPO_DIR)
    url = REMOTE_URL
    if token:
        # Embed token in HTTPS url
        url = f"https://{token}@github.com/Ayushdwivedi2327/cyberMirage.git"
        print("[+] Using authenticated GitHub token...")

    print(f"[*] Pushing local branch to {REMOTE_URL} (branch: main / master)...")
    try:
        # Push to remote main branch
        porcelain.push(repo, url, "HEAD:refs/heads/main")
        print("[SUCCESS] Pushed successfully to refs/heads/main!")
    except Exception as e:
        print(f"[!] Push to main failed: {e}")
        try:
            porcelain.push(repo, url, "HEAD:refs/heads/master")
            print("[SUCCESS] Pushed successfully to refs/heads/master!")
        except Exception as e2:
            print(f"[!] Push to master failed: {e2}")
            print("\nNote: To push to GitHub over HTTPS, GitHub requires a Personal Access Token (PAT) with 'repo' scope.")
            print("You can run: python scripts/git_push.py <YOUR_GITHUB_TOKEN>")


if __name__ == "__main__":
    token_arg = None
    if len(sys.argv) > 1:
        token_arg = sys.argv[1]
    elif os.environ.get("GITHUB_TOKEN"):
        token_arg = os.environ.get("GITHUB_TOKEN")
    else:
        try:
            token_arg = input("Enter your GitHub Personal Access Token (or press Enter to try unauthenticated): ").strip()
            if not token_arg:
                token_arg = None
        except Exception:
            token_arg = None
    push_repo(token_arg)
