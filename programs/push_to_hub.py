"""
push_to_hub.py - Step 6: upload the fine-tuned model to the Hugging Face Hub.

The Streamlit app loads the model from config.HF_MODEL_REPO, so the app
always uses exactly the model that finetune.py produced.

Needs a Hugging Face access token with "write" permission. Log in once with
`hf auth login` (or set the HF_TOKEN environment variable); the token is a
secret and is never written into this repository.

Usage
  python push_to_hub.py
"""

import argparse

from huggingface_hub import HfApi

import config
import utils


def main() -> None:
    model_dir = utils.finetuned_dir(smoke=False)
    if not model_dir.exists():
        raise SystemExit(f"{model_dir} not found - run finetune.py first.")
    if config.HF_MODEL_REPO.startswith("YOUR-HF-USERNAME"):
        raise SystemExit("Set HF_MODEL_REPO in config.py to your own Hugging Face username first.")

    api = HfApi()
    try:
        user = api.whoami()["name"]
    except Exception as err:
        raise SystemExit(f"Not logged in to Hugging Face ({err}). Run `hf auth login` first.") from err

    print(f"Uploading {model_dir} -> {config.HF_MODEL_REPO} as {user}")
    try:
        api.create_repo(config.HF_MODEL_REPO, repo_type="model", exist_ok=True)
        api.upload_folder(folder_path=str(model_dir), repo_id=config.HF_MODEL_REPO,
                          commit_message="Upload fine-tuned scam classifier")
    except Exception as err:
        raise SystemExit(f"Upload failed: {err}") from err
    print(f"Done: https://huggingface.co/{config.HF_MODEL_REPO}")


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    main()
