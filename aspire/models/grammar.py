"""Grammar-constrained decoding baseline (Geng et al., EMNLP 2023) via transformers-cfg.

The grammar covers the ASP subset of the test suite (normal and disjunctive
rules, integrity constraints, weak constraints, default negation, comparisons)
plus facts. Predicate names are any lowercase identifier: the constraint only
guarantees syntactic well-formedness, not the choice of predicates or conditions.
Ported from the repository's earlier asp_constrained_decoding.py (static mode).
"""

ASP_GRAMMAR = r"""
root ::= ws rule (ws rule)* ws
rule ::= normalrule | fact | constraint | weakconstraint
normalrule ::= head ws ":-" ws body ws "."
fact ::= head ws "."
constraint ::= ":-" ws body ws "."
weakconstraint ::= ":~" ws body ws "." ws "[" ws term ws "@" ws term (ws "," ws term)* ws "]"
head ::= atom (ws "|" ws atom)*
body ::= literal (ws "," ws literal)*
literal ::= naf | comparison | atom
naf ::= "not" sp atom
comparison ::= term ws cmpop ws term
cmpop ::= ">=" | "<=" | "==" | "!=" | ">" | "<" | "="
atom ::= predname (ws "(" ws term (ws "," ws term)* ws ")")?
term ::= variable | number | string | predname | "_"
predname ::= [a-z] idchar*
variable ::= [A-Z] idchar*
idchar ::= [a-zA-Z0-9_]
number ::= "-"? [0-9]+
string ::= "\"" strchar* "\""
strchar ::= [^"\n]
sp ::= " " ws
ws ::= wschar*
wschar ::= [ \t\n]
"""


def processor_factory(tokenizer, grammar=ASP_GRAMMAR):
    """Zero-argument factory returning a fresh logits processor per generate() call.

    The byte trie over the vocabulary is built once; the parsing state is reset
    so that it does not leak across samples.
    """
    from transformers_cfg.generation.logits_process import GrammarConstrainedLogitsProcessor
    from transformers_cfg.grammar_utils import IncrementalGrammarConstraint

    constraint = IncrementalGrammarConstraint(grammar, "root", tokenizer)

    def factory():
        constraint.reset()
        return GrammarConstrainedLogitsProcessor(constraint)

    return factory
