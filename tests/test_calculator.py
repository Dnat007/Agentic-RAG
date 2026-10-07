import pytest

from app.tools.calculator import CalculatorError, SafeCalculator


@pytest.fixture
def calculator():
    return SafeCalculator()


def test_addition(calculator):
    assert calculator.calculate("10 + 20") == 30


def test_subtraction(calculator):
    assert calculator.calculate("20 - 5") == 15


def test_multiplication(calculator):
    assert calculator.calculate("10 * 5") == 50


def test_division(calculator):
    assert calculator.calculate("100 / 4") == 25


def test_modulo(calculator):
    assert calculator.calculate("10 % 3") == 1


def test_power(calculator):
    assert calculator.calculate("2 ** 10") == 1024


def test_parentheses(calculator):
    assert calculator.calculate("(10 + 5) * 2") == 30


def test_float_expression(calculator):
    assert calculator.calculate("10.5 * 2") == 21.0


def test_negative_number(calculator):
    assert calculator.calculate("-10 + 5") == -5


def test_operator_precedence(calculator):
    assert calculator.calculate("10 + 5 * 2") == 20


def test_nested_expression(calculator):
    assert calculator.calculate("((10 + 5) * 2) / 5") == 6


def test_empty_expression(calculator):
    with pytest.raises(CalculatorError):
        calculator.calculate("")


def test_invalid_expression(calculator):
    with pytest.raises(CalculatorError):
        calculator.calculate("10 +")


def test_unsupported_function(calculator):
    with pytest.raises(CalculatorError):
        calculator.calculate("sqrt(16)")


def test_unsupported_string(calculator):
    with pytest.raises(CalculatorError):
        calculator.calculate("'hello'")


def test_eval_injection_is_blocked(calculator):
    with pytest.raises(CalculatorError):
        calculator.calculate("__import__('os').system('whoami')")


def test_boolean_is_blocked(calculator):
    with pytest.raises(CalculatorError):
        calculator.calculate("True")


def test_expression_length_limit(calculator):
    with pytest.raises(CalculatorError):
        calculator.calculate("1" * 501)


def test_large_exponent_is_blocked(calculator):
    with pytest.raises(CalculatorError):
        calculator.calculate("2 ** 1001")


def test_non_string_input(calculator):
    with pytest.raises(CalculatorError):
        calculator.calculate(123)