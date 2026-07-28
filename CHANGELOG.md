# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Docker containerization with multi-stage builds
- GitHub Actions CI/CD pipeline (test → build → deploy)
- GitHub Container Registry (ghcr.io) image publishing
- Remote `.env` management via `DEPLOY_ENV_FILE` secret
- SSL/TLS support via nginx
- Issue templates, PR template, CODEOWNERS, Code of Conduct

### Changed
- Migrated from GitLab CI to GitHub Actions
- Switched from systemd-based deployment to Docker Compose
- LLM backend changed from BayesDL (SSH tunnel) to DashScope direct API
- Health check uses Python urllib instead of curl

### Removed
- GitLab-specific CI/CD configuration
- systemd service files for agent-backend and nginx
- Dependabot branch clutter on remote