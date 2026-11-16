# Publish Snowline manually

The source repository is [Weiyu-USTC/Snowline](https://github.com/Weiyu-USTC/Snowline). The README is prepared for the `v0.1.0` release. Its download links become available after the release and both assets below are published.

## Push the source with GitHub Desktop

1. Use **File → Add Local Repository** and select the `snowline-github` folder containing this README. If it is already listed, select that repository.
2. On `main`, review the changes, enter a commit summary, and click **Commit to main**.
3. Click **Push origin**. Review the repository's Actions results after pushing. The preceding source revision passed both the Linux Python checks and the macOS widget build in [this run](https://github.com/Weiyu-USTC/Snowline/actions/runs/43799532516).

The runtime ZIP is separate from source control. Do not force-add downloaded data, weights, environments, or compiled apps to Git.

## Publish the model bundle

1. Open the repository's **Releases** page yourself and choose **Draft a new release**.
2. Create tag **`v0.1.0`** targeting the `main` commit you just pushed.
3. Set the title to **Snowline v0.1.0 — model bundle**. Describe it as the frozen research inference bundle for Alta, Vail, Stowe, and Jay Peak; link to the README installation instructions and `docs/DATA_SOURCES.md`.
4. Attach **`snowline-runtime-2026-11-15.zip`** (46,633,267 bytes) and **`SHA256SUMS.txt`**. Use the checksum file prepared specifically for this release, whose only entry is shown below.
5. Wait for both uploads to finish, then publish the release.
6. Open the README's release and asset links to confirm they download successfully. GitHub automatically supplies source archives for the release tag.

Expected `SHA256SUMS.txt` contents:

```text
aa6401bb0b91ba3f94f084355930deb73e172bde2768ff5b93e7ec67d073611a  snowline-runtime-2026-11-15.zip
```

Keep the tag and asset names exactly as written so the README links work. Keep matching source and runtime versions together: the installer verifies all 126 files against the hashes in `artifacts/runtime-files.json`. Uploading a release asset is a separate step from pushing commits with GitHub Desktop.

## Release contents

The source archive is intentionally usable without the original author's machine paths or working directory. The companion bundle contains frozen inference models and required reference/report files; it excludes the bulk training archives. The installer verifies all files before installation. Model serialization can execute Python code when loaded, so only install the bundle matching the trusted source release and manifest.

Current forecasts are downloaded on demand. The local web service binds to `127.0.0.1`; GitHub Pages cannot host its Python/model backend. Running the dashboard locally or designing a separately secured server deployment is required. The floating app is built locally and is ad-hoc signed, not notarized for App Store distribution.

## Licensing

This export preserves attribution notices and does not choose a new source-code license for you. A public repository is not automatically an open-source license. Source, observations, pretrained third-party weights, and locally fitted artifacts are distinct materials; retain their applicable notices. Alta observation data carries CC BY-NC 3.0 terms. Consult `DATA_SOURCES.md` before deciding how to redistribute the companion bundle or use it commercially.
