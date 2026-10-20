"""
Logical Reasoning - Deductive and Inductive Logic
"""
from dataclasses import dataclass
from typing import List, Dict, Optional, Any, Set
from enum import Enum


class LogicalOperator(Enum):
    AND = "AND"
    OR = "OR"
    NOT = "NOT"
    IMPLIES = "IMPLIES"
    IFF = "IFF"


@dataclass
class Proposition:
    symbol: str
    value: Optional[bool] = None
    description: str = ""


class PropositionalLogic:
    def __init__(self):
        self.propositions: Dict[str, Proposition] = {}
    
    def add_proposition(self, symbol: str, description: str = "", value: Optional[bool] = None):
        prop = Proposition(symbol=symbol, description=description, value=value)
        self.propositions[symbol] = prop
        return prop
    
    def evaluate(self, expr: Any, context: Optional[Dict[str, bool]] = None) -> bool:
        context = context or {}
        context.update({p: v.value for p, v in self.propositions.items() if v.value is not None})
        return self._eval(expr, context)
    
    def _eval(self, expr: Any, context: Dict[str, bool]) -> bool:
        if isinstance(expr, str):
            return context.get(expr, False)
        elif isinstance(expr, list) and len(expr) > 0:
            op = expr[0]
            if op == LogicalOperator.NOT:
                return not self._eval(expr[1], context)
            elif op == LogicalOperator.AND:
                return all(self._eval(a, context) for a in expr[1:])
            elif op == LogicalOperator.OR:
                return any(self._eval(a, context) for a in expr[1:])
            elif op == LogicalOperator.IMPLIES:
                return (not self._eval(expr[1], context)) or self._eval(expr[2], context)
            elif op == LogicalOperator.IFF:
                return self._eval(expr[1], context) == self._eval(expr[2], context)
        return False


class FirstOrderLogic:
    pass


class LogicalDeduction:
    def __init__(self):
        self.prop_logic = PropositionalLogic()
        self.first_order = FirstOrderLogic()
