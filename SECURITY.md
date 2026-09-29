# Security reports

Report suspected vulnerabilities privately to the repository maintainers. Use GitHub's private vulnerability reporting option if the repository has enabled it; otherwise ask a maintainer for a private reporting channel without disclosing exploit details in a public issue.

Include the affected version, reproduction steps, expected and actual behavior, and a minimal synthetic example. Remove access tokens, private addresses, recordings and personal data from logs and screenshots.

AgentX currently uses one trusted team workspace and a shared API token. Keep services on loopback during local development. For network access, configure authentication and HTTPS as described in [deployment](docs/DEPLOYMENT.md). Do not use a shared deployment for mutually untrusted tenants.

If a credential has been exposed, revoke or rotate it first. Deleting a file or adding it to an ignore file does not remove its contents from previous commits, forks, downloaded archives or logs.
