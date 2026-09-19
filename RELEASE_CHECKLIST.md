# v1.0.0 release checklist

## Repository readiness

- [x] Package and runtime versions set to `1.0.0`.
- [x] Complete benchmark workflow implemented without provider-specific code.
- [x] Configuration, lifecycle, artifact, and secret-redaction checks enabled.
- [x] Environment and reproducibility metadata captured for completed runs.
- [x] README, architecture, configuration, usage, workflow, and Docker docs added.
- [x] CPU-only Dockerfile uses an unprivileged user and bundles no models.
- [x] Full unit, integration, CLI, and mocked end-to-end suite passes.

## Release operator actions

- [x] Build a clean local `1.0.0` wheel from `pyproject.toml`.
- [ ] Build the source distribution in the release environment.
- [ ] Install the wheel into a fresh Python 3.11 environment and run smoke tests.
- [ ] Install the wheel into a fresh Python 3.12 environment and run smoke tests.
- [ ] Build the Docker image in the release environment.
- [ ] Run the image against approved staging endpoints and mounted data/output.
- [ ] Review generated artifacts and confirm no organizational secrets or PII.
- [ ] Create and sign the `v1.0.0` source-control tag.
- [ ] Publish approved Python and container artifacts.

Release publication and cloud-specific deployment remain operator-controlled
activities and are not performed by the benchmark pipeline.
