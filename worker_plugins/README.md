# Worker Plugins

Packages in this directory run in dependency-isolated worker environments.
They are kept outside the core `src/rosclaw` package so optional robotics and
machine-learning stacks cannot change the core CLI's Python or dependency
contract.

Each plugin owns its source, packaging metadata, and plugin-specific fixtures.
The core distribution may bundle selected fixtures, but core processes must
not import heavyweight worker frameworks during discovery or status checks.
