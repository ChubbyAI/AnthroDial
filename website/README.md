# AnthroDial website

Public website: https://chubbyai.github.io/AnthroDial/

This directory contains the standalone static website. Open `index.html` in a browser to preview it locally. No build step or dependencies are needed. `release-results.json` provides the downloadable result summaries used in the original website release.

The GitHub Actions workflow in `.github/workflows/pages.yml` deploys only this directory to GitHub Pages when website files change on `main`. It can also be started manually from the Actions tab. The repository Pages publishing source is **GitHub Actions**.

To update the website, replace the files in this directory and push to `main`. Keep embedded leaderboard data in `index.html` consistent with the downloadable summaries. Repository code, results, and documentation outside this directory are not part of the website deployment.
