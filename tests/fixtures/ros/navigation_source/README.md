# Installed navigation source fixtures

These are unchanged bytes read from the frozen generic runtime image, used to check generic parameter generation against actual installed Nav2 and OpenNav templates. `source-manifest.json` records their original paths, image digest, package version and SHA-256 hashes. Both packages declare Apache-2.0; the installed package metadata, Nav2 copyright notice and OpenNav license are retained.

The installed OpenNav source tree has no `.git`; its revision cannot be independently obtained through `git rev-parse` in this image. The template hashes identify the tested source bytes. Synthetic robot XML is used by tests. No held-out robot was selected, and these fixtures do not prove controller startup, graph readiness, navigation or physical cleaning.
