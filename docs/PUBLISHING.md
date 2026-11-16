# Put Snowline in your GitHub repository

1. Extract `snowline-source-2026-11-15.zip`.
2. Copy the **contents** of its `snowline/` folder into your intended repository. For an existing project, review changes before replacing files; retain its applicable license and repository settings.
3. Commit the source, docs, workflow, requirements, and artifact manifest. `.gitignore` excludes runtime data, checkpoints, caches, environments, and built apps.
4. Create a GitHub Release and attach `snowline-runtime-2026-11-15.zip` and the supplied `SHA256SUMS.txt`. The model bundle belongs in release assets, not ordinary Git history.
5. Update the README with your release download link. Keep matching source and runtime versions together; the installer checks the per-file hashes in the source manifest.
6. Review the GitHub Actions results after pushing. The included workflow was tested locally in its component checks; it has not been run in your GitHub repository yet.

For a new empty repository, after configuring your own remote:

```sh
git init
git add .
git status --short
git commit -m "Prepare Snowline research app"
git branch -M main
git remote add origin YOUR_REPOSITORY_URL
git push -u origin main
```

Use your actual repository URL in place of `YOUR_REPOSITORY_URL`. For an existing checkout, use its current branch and remote rather than re-running these initialization steps. This package does not create a repository, upload files, publish a release, or alter access settings on your behalf.

## Release contents

The source archive is intentionally usable without the original author's machine paths or working directory. The companion bundle contains frozen inference models and required reference/report files; it excludes the bulk training archives. The installer verifies all files before installation. Model serialization can execute Python code when loaded, so only install the bundle matching the trusted source release and manifest.

Current forecasts are downloaded on demand. The local web service binds to `127.0.0.1`; GitHub Pages cannot host its Python/model backend. Running the dashboard locally or designing a separately secured server deployment is required. The floating app is built locally and is ad-hoc signed, not notarized for App Store distribution.

## Licensing

This export preserves attribution notices and does not choose a new source-code license for you. A public repository is not automatically an open-source license. Source, observations, pretrained third-party weights, and locally fitted artifacts are distinct materials; retain their applicable notices. Alta observation data carries CC BY-NC 3.0 terms. Consult `DATA_SOURCES.md` before deciding how to redistribute the companion bundle or use it commercially.
