# AnthroDial website

Intended website URL: https://chubbyai.github.io/AnthroDial/ (requires GitHub Pages to be enabled).

This directory contains the standalone static website. Open `index.html` in a browser to preview it locally. No build step or dependencies are needed. `release-results.json` provides the downloadable result summaries used in the original website release.

The GitHub Actions workflow in `.github/workflows/pages.yml` deploys only this directory to GitHub Pages. After enabling Pages and setting its publishing source to **GitHub Actions**, start the workflow manually from the Actions tab. It is currently manual because this private repository does not support Pages under the current account plan.

To update the website, replace the files in this directory, push to `main`, and run the deployment workflow after Pages is enabled. Keep embedded leaderboard data in `index.html` consistent with the downloadable summaries. Repository code, results, and documentation outside this directory are not part of the website deployment.

GitHub Pages from a private repository requires a supported GitHub plan. The intended website is publicly accessible once deployed.
