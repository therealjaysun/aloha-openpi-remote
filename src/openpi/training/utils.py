"""Small checkpoint-conversion helpers retained by the inference workflow."""

from collections.abc import Callable
from typing import Any

import jax

from openpi.shared import array_typing as at


@at.typecheck
def tree_to_info(tree: at.PyTree, interp_func: Callable[[Any], str] = str) -> str:
    tree, _ = jax.tree_util.tree_flatten_with_path(tree)
    return "\n".join(f"{jax.tree_util.keystr(path)}: {interp_func(value)}" for path, value in tree)


@at.typecheck
def array_tree_to_info(tree: at.PyTree) -> str:
    return tree_to_info(tree, lambda x: f"{x.shape}@{x.dtype}")
