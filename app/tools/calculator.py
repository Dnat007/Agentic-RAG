from __future__ import annotations

import ast
import math
import operator
from typing import Union


Number = Union[int, float]


class CalculatorError(ValueError):
    """Raised when a mathematical expression is invalid or unsafe."""


class SafeCalculator:
    """
    Safe deterministic calculator.

    Supported:
        +, -, *, /, %, **
        unary +, unary -
        parentheses
        integer and floating-point numbers

    Examples:
        25 * 48
        (100 + 50) / 5
        2 ** 10
        -10 + 5
    """

    _binary_operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
    }

    _unary_operators = {
        ast.UAdd: operator.pos,
        ast.USub: operator.neg,
    }

    def calculate(self, expression: str) -> Number:
        if not isinstance(expression, str):
            raise CalculatorError("Expression must be a string.")

        expression = expression.strip()

        if not expression:
            raise CalculatorError("Expression cannot be empty.")

        if len(expression) > 500:
            raise CalculatorError("Expression is too long.")

        try:
            tree = ast.parse(expression, mode="eval")
        except SyntaxError as exc:
            raise CalculatorError("Invalid mathematical expression.") from exc

        try:
            result = self._evaluate(tree.body)
        except CalculatorError:
            raise
        except (ArithmeticError, OverflowError, ValueError) as exc:
            raise CalculatorError(str(exc)) from exc

        if isinstance(result, float):
            if not math.isfinite(result):
                raise CalculatorError("Result is not finite.")

        return result

    def _evaluate(self, node: ast.AST) -> Number:
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool):
                raise CalculatorError("Boolean values are not supported.")

            if isinstance(node.value, (int, float)):
                if isinstance(node.value, float) and not math.isfinite(node.value):
                    raise CalculatorError("Non-finite numbers are not supported.")

                return node.value

            raise CalculatorError("Only numeric constants are supported.")

        if isinstance(node, ast.BinOp):
            operation = self._binary_operators.get(type(node.op))

            if operation is None:
                raise CalculatorError(
                    f"Operator '{type(node.op).__name__}' is not supported."
                )

            left = self._evaluate(node.left)
            right = self._evaluate(node.right)

            if isinstance(node.op, ast.Pow):
                if abs(right) > 1000:
                    raise CalculatorError("Exponent is too large.")

            result = operation(left, right)

            if isinstance(result, float) and not math.isfinite(result):
                raise CalculatorError("Result is not finite.")

            return result

        if isinstance(node, ast.UnaryOp):
            operation = self._unary_operators.get(type(node.op))

            if operation is None:
                raise CalculatorError(
                    f"Unary operator '{type(node.op).__name__}' is not supported."
                )

            return operation(self._evaluate(node.operand))

        raise CalculatorError(
            f"Expression element '{type(node).__name__}' is not supported."
        )