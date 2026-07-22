# inference-engine

`context_resolver` is a small Python library for building and querying a typed,
demand-driven semantic tree. It is designed for applications that need to keep a
structured context, resolve missing values on demand, and invalidate cached
results when dependencies change.

## What It Is

The library models application state as a tree of typed nodes. Nodes can be
fully specified, partially specified, or underspecified. When you query a path,
the `Context` can resolve unresolved nodes lazily through a configured
resolution pipeline.

The project is useful when you want:

* typed tree structures instead of ad hoc dictionaries
* lazy resolution of missing values
* dependency tracking and cache invalidation
* serialization, templating, and mockable resolution strategies

## How To Use The Library

Install the package in editable mode while developing:

```bash
python -m pip install -e .
```

Build a tree with nodes, then query it through a `Context`:

```python
from context_resolver import Context, Path
from context_resolver.ast.nodes import MappingNode, ScalarNode

root = MappingNode({
	"name": ScalarNode("Alice"),
	"age": ScalarNode(30),
})

ctx = Context(root)

name_node = ctx.query(Path("name"))
print(name_node.value)

ctx.set(Path("name"), ScalarNode("Bob"))
print(ctx.get_value(Path("name")))
```

For lazy resolution, create a `Resolver` with a template registry and one or
more passes, then attach it to the `Context`. The example in
[`context_resolver/examples/simple_prompt.py`](context_resolver/examples/simple_prompt.py)
shows the full flow end to end.

## Top-Level API

The package root exports the main entry points for most applications:

* `context_resolver.Node` - abstract base class for all nodes
* `context_resolver.NodeState` - node completeness state enum
* `context_resolver.Path` - typed path object used to address nodes
* `context_resolver.Context` - primary interface for querying and mutating the tree

Common supporting APIs live in subpackages:

* `context_resolver.ast` - node types, schema types, and path helpers
* `context_resolver.query` - `Resolver`, passes, and dependency tracking
* `context_resolver.templates` - prompt templates and registries
* `context_resolver.inference` - resolution provider and strategy interfaces
* `context_resolver.serialization` - context serialization and diff helpers

The most commonly used concrete node classes are:

* `context_resolver.ast.nodes.ScalarNode`
* `context_resolver.ast.nodes.MappingNode`
* `context_resolver.ast.nodes.SequenceNode`
* `context_resolver.ast.resolvable_node.ResolvableNode`

For lazy resolution workflows, the usual setup is:

1. Build a tree of nodes.
2. Register templates in a `TemplateRegistry`.
3. Configure a `Resolver` with one or more passes.
4. Query the `Context` with a `Path`.
5. Mutate dependencies with `Context.set()` and let invalidation happen automatically.

## Repository Notes

Run the test suite with:

```bash
python -m pytest
```
