"""Factor expression trees: parse / evaluate / random generation / mutation helpers."""
import ast

import ops

# Legacy / shorthand names -> phandas-modify operator names.
ALIASES = {
    "neg": "reverse",
    "sub": "subtract",
    "mul": "multiply",
    "div": "divide",
    "log": "s_log_1p",
    "log1p_abs": "s_log_1p",
    "cs_rank": "rank",
    "cs_zscore": "zscore",
    "delay": "ts_delay",
    "delta": "ts_delta",
    "ts_std": "ts_std_dev",
}


class Node:
    __slots__ = ("kind", "name", "children", "value")

    def __init__(self, kind, name=None, children=None, value=None):
        self.kind = kind  # 'call' | 'terminal' | 'window'
        self.name = name
        self.children = children or []
        self.value = value

    def to_str(self):
        if self.kind == "terminal":
            return self.name
        if self.kind == "window":
            return str(int(self.value))
        return f"{self.name}({', '.join(c.to_str() for c in self.children)})"

    def copy(self):
        return Node(self.kind, self.name, [c.copy() for c in self.children], self.value)


def nodes(tree):
    out = [tree]
    for c in tree.children:
        out.extend(nodes(c))
    return out


def complexity(tree):
    return len(nodes(tree))


def depth(tree):
    if not tree.children:
        return 1
    return 1 + max(depth(c) for c in tree.children)


def has_windows(tree):
    return any(n.kind == "window" for n in nodes(tree))


def scale_windows(tree, k):
    t = tree.copy()
    for n in nodes(t):
        if n.kind == "window":
            n.value = max(2, int(round(n.value * k)))
    return t


def evaluate(tree, data):
    if tree.kind == "terminal":
        return data[tree.name]
    if tree.kind == "window":
        return int(tree.value)
    fn = ops.OPS[tree.name]
    return fn(*[evaluate(c, data) for c in tree.children])


def validate(tree):
    if tree.kind == "window":
        raise ValueError("bare integer not allowed here (windows only as 2nd arg of ts ops)")
    if tree.kind == "terminal":
        if tree.name not in ops.TERMINALS:
            raise ValueError(f"unknown terminal: {tree.name}")
        return
    if tree.name in ops.BINARY:
        if len(tree.children) != 2:
            raise ValueError(f"{tree.name} needs 2 args")
        for c in tree.children:
            validate(c)
    elif tree.name in ops.UNARY:
        if len(tree.children) != 1:
            raise ValueError(f"{tree.name} needs 1 arg")
        validate(tree.children[0])
    elif tree.name in ops.TS:
        if len(tree.children) != 2 or tree.children[1].kind != "window":
            raise ValueError(f"{tree.name} needs (series, integer_window)")
        if int(tree.children[1].value) < 2:
            raise ValueError("window must be >= 2")
        validate(tree.children[0])
    else:
        raise ValueError(f"unknown op: {tree.name}")


def parse(expr_str):
    t = ast.parse(expr_str, mode="eval")

    def conv(n):
        if isinstance(n, ast.Expression):
            return conv(n.body)
        if isinstance(n, ast.Call):
            if not isinstance(n.func, ast.Name):
                raise ValueError("only simple function calls allowed")
            name = ALIASES.get(n.func.id, n.func.id)
            if name not in ops.OPS:
                raise ValueError(f"unknown op: {n.func.id}")
            return Node("call", name, [conv(a) for a in n.args])
        if isinstance(n, ast.Name):
            return Node("terminal", n.id)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return Node("window", value=int(n.value))
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub):
            inner = conv(n.operand)
            if inner.kind == "window":
                inner.value = -inner.value
                return inner
            return Node("call", "reverse", [inner])
        raise ValueError(f"unsupported syntax: {ast.dump(n)}")

    tree = conv(t)
    validate(tree)
    return tree


def random_expr(rng, max_depth=4, _depth=1):
    """rng: random.Random instance."""
    if _depth >= max_depth or (_depth > 1 and rng.random() < 0.3):
        return Node("terminal", rng.choice(ops.TERMINALS))
    r = rng.random()
    if r < 0.35:
        name = rng.choice(ops.BINARY)
        return Node("call", name, [random_expr(rng, max_depth, _depth + 1),
                                   random_expr(rng, max_depth, _depth + 1)])
    if r < 0.6:
        name = rng.choice(ops.UNARY)
        return Node("call", name, [random_expr(rng, max_depth, _depth + 1)])
    name = rng.choice(ops.TS)
    return Node("call", name, [random_expr(rng, max_depth, _depth + 1),
                               Node("window", value=rng.choice(ops.WINDOWS))])
