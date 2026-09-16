# Publication privacy review

Review date: 16 September 2026. Scope: this guide repository and its available published history, not the upstream projects or the owner's wider GitHub account.

## Release policy

The public package contains documentation, runtime integration code, numerical tests, source identifiers, and sanitized aggregate measurements. It must not contain personal names of the deployment operator, private addresses or hostnames, personal computer paths, account credentials, API tokens, SSH key material, filled deployment profiles, raw operational logs, or conversations.

Upstream project names and credits, public repository URLs, software paths inside containers, and cryptographic hashes identifying public or documented artifacts are intentional. They are needed for attribution and reproduction. Loopback addresses refer to the reader's own machine; network and account fields in the current configuration template require the reader's own values.

## Scope of checks

- Read the release files and inspect all reachable historical file versions, filenames, commit messages, and author/committer metadata.
- Search for known private deployment identifiers, personal account paths, credentials, common token formats, key blocks, and SSH key material; inspect address and email candidates rather than treating every hash or version number as a secret.
- Inspect GitHub branch/ref inventory, repository description, releases, issues, commit comments, and workflow artifacts for additional published material.
- Remove desktop-client-specific setup and measurements from the current release; retain generic API instructions.
- Replace illustrative network/account constants with explicit placeholders and expand ignore rules for local profiles, private key files, database sidecars, and backups.

The pre-publication scan covered **22 candidate release files**, both previously published commits through `ad82d3c`, and **23 distinct historical file blobs**. The remote had only its `main` branch, no releases, issues, issue/commit comments, workflow runs, or workflow artifacts. Discussions and Pages were disabled; the wiki URL redirected to the repository, with no published wiki found.

Documentation checks covered **24 Bash blocks**, three embedded Python snippets, and local Markdown links. The profile merge, cache update, and per-rank layer validator also ran against disposable fixtures, including incomplete profiles, an existing configuration file, missing fired layers, and a wrong alpha. These were local checks, not a fresh model deployment. The runtime suffix's SHA-256 remained unchanged.

## Findings and limits

The completed review found no deployment operator's personal name, real private endpoint, personal workstation path, API credential, or SSH key material in the inspected publication. Historical examples used fictional network/account values; the current template uses explicit placeholders. The historical desktop-client references remain in earlier commits, not in the current guide. No sensitive-history removal was required by these findings.

The repository remains associated with its public GitHub owner. Existing commit attribution uses the account handle and a GitHub noreply address. File sanitization does not anonymize the owner, account profile, upstream authors, or third-party copies of an earlier public revision. The review does not certify material outside this repository or guarantee detection of every possible secret format.

The checks were performed without starting, stopping, or reconfiguring the model servers. Future releases need a new review: `.gitignore` reduces accidental inclusion but does not prevent a force-add, sanitize logs, or remove anything already committed.

Keep your filled profile, `.env`, credentials, downloaded source artifacts, build outputs, and diagnostic logs local. Before posting a support excerpt, replace your own addresses, hostnames, usernames, paths, and authentication values with descriptive placeholders.
